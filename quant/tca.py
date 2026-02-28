"""
Advanced Transaction Cost Analysis (TCA) — Institutional-Grade Execution Analytics

Measures three dimensions of execution quality:

1. Market Impact: Did our order push the micro-price against us before filling?
   - Pre-trade micro-price vs fill price (immediate impact)
   - Temporary impact: price 10s after fill vs fill price (mean reversion = our impact)

2. Opportunity Cost (Missed Alpha): When we skip a trade (correlation guard, max positions,
   earnings block), how much profit did we leave on the table?
   - Tracks skipped signals and measures 1h/4h/EOD price movement in skipped direction

3. Adverse Selection: What happens to the price AFTER our fill?
   - 1-second, 10-second, 60-second post-fill price movement
   - If price consistently drops after our buys → we're being front-run
   - Tracks per-symbol adverse selection score over rolling 30-trade window

All metrics logged to Supabase table `tca_metrics` for dashboard analysis.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict, deque

from referee.secret_vault import get_secret
from supabase import create_client, Client
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Rolling adverse selection tracker (per-symbol, last 30 fills)
_adverse_selection: dict = defaultdict(lambda: deque(maxlen=30))

# Skipped signal tracker for opportunity cost
_skipped_signals: list = []


# ── 1. Market Impact ─────────────────────────────────────────────────────────

def measure_market_impact(
    symbol: str,
    side: str,
    pre_trade_micro: float,
    fill_price: float,
    fill_qty: int,
) -> dict:
    """
    Measure immediate market impact of our execution.

    Returns:
      - immediate_impact_bps: (fill_price - pre_trade_micro) / micro * 10000
        Positive = we paid more than fair value (adverse for buys)
      - signed_impact_bps: direction-adjusted (positive = bad for us)
    """
    if pre_trade_micro <= 0:
        return {"immediate_impact_bps": 0.0, "signed_impact_bps": 0.0}

    raw_impact = (fill_price - pre_trade_micro) / pre_trade_micro * 10000

    # Direction-adjust: for buys, paying more = bad; for sells, receiving less = bad
    if side == "buy":
        signed = raw_impact  # positive = overpaid
    else:
        signed = -raw_impact  # positive = undersold

    return {
        "immediate_impact_bps": round(raw_impact, 2),
        "signed_impact_bps": round(signed, 2),
        "pre_trade_micro": round(pre_trade_micro, 4),
        "fill_price": round(fill_price, 4),
        "fill_qty": fill_qty,
    }


# ── 2. Adverse Selection Tracking ────────────────────────────────────────────

def schedule_adverse_selection_check(
    symbol: str,
    side: str,
    fill_price: float,
    fill_time: datetime,
    alpaca_order_id: str,
):
    """
    Schedule background price checks at 1s, 10s, and 60s after fill.
    Runs in a daemon thread so it doesn't block the main cycle.
    """
    def _check():
        results = {"symbol": symbol, "side": side, "fill_price": fill_price,
                   "alpaca_order_id": alpaca_order_id}

        alpaca_sym = "TSM" if symbol == "TSMC" else symbol

        for delay_label, delay_sec in [("1s", 1), ("10s", 10), ("60s", 60)]:
            time.sleep(delay_sec if delay_label == "1s" else (delay_sec - (1 if delay_label == "10s" else 10)))
            try:
                req = StockLatestQuoteRequest(symbol_or_symbols=alpaca_sym, feed=DataFeed.IEX)
                quote = data_client.get_stock_latest_quote(req)
                bid = float(quote[alpaca_sym].bid_price or 0)
                ask = float(quote[alpaca_sym].ask_price or 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                else:
                    mid = fill_price

                # Price movement since fill
                pct_move = (mid - fill_price) / fill_price

                # Direction-adjust: for buys, price dropping = adverse; for sells, price rising = adverse
                if side == "buy":
                    adverse_bps = -pct_move * 10000  # positive = price dropped (bad for buy)
                else:
                    adverse_bps = pct_move * 10000   # positive = price rose (bad for sell)

                results[f"price_{delay_label}"] = round(mid, 4)
                results[f"adverse_{delay_label}_bps"] = round(adverse_bps, 2)

            except Exception as e:
                results[f"price_{delay_label}"] = None
                results[f"adverse_{delay_label}_bps"] = None

        # Log results
        adverse_1s = results.get("adverse_1s_bps", 0) or 0
        adverse_10s = results.get("adverse_10s_bps", 0) or 0
        adverse_60s = results.get("adverse_60s_bps", 0) or 0

        # Track rolling adverse selection per symbol
        _adverse_selection[symbol].append({
            "1s": adverse_1s, "10s": adverse_10s, "60s": adverse_60s,
            "ts": datetime.now(timezone.utc),
        })

        # Alert if consistently adverse
        recent = list(_adverse_selection[symbol])
        if len(recent) >= 5:
            avg_60s = np.mean([r["60s"] for r in recent[-5:]])
            if avg_60s > 3.0:  # >3bps average adverse selection over last 5 fills
                print(f"[TCA] ⚠ {symbol}: ADVERSE SELECTION ALERT — avg 60s adverse={avg_60s:.1f}bps over last 5 fills")
                results["adverse_alert"] = True

        print(f"[TCA] {symbol} post-fill: 1s={adverse_1s:+.1f}bps 10s={adverse_10s:+.1f}bps 60s={adverse_60s:+.1f}bps")

        # Write to Supabase
        _log_tca_metric("adverse_selection", results)

    thread = threading.Thread(target=_check, daemon=True, name=f"tca_{symbol}")
    thread.start()


# ── 3. Opportunity Cost (Missed Alpha) ───────────────────────────────────────

def log_skipped_signal(
    symbol: str,
    direction: str,
    skip_reason: str,
    ensemble_score: float,
    mid_price: float,
):
    """
    Record a signal that was generated but not executed.
    The opportunity cost checker will measure what happened to the price.
    """
    _skipped_signals.append({
        "symbol": symbol,
        "direction": direction,
        "skip_reason": skip_reason,
        "ensemble_score": ensemble_score,
        "mid_price": mid_price,
        "ts": datetime.now(timezone.utc),
    })
    # Cap at 200 entries
    if len(_skipped_signals) > 200:
        del _skipped_signals[:100]


def measure_opportunity_cost():
    """
    For all skipped signals in the last 4 hours, measure how much alpha was missed.
    Called by the labeler background task.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
    recent = [s for s in _skipped_signals if s["ts"] > cutoff]

    if not recent:
        return

    print(f"[TCA] Measuring opportunity cost for {len(recent)} skipped signals...")

    for signal in recent:
        symbol = signal["symbol"]
        direction = signal["direction"]
        entry_price = signal["mid_price"]
        alpaca_sym = "TSM" if symbol == "TSMC" else symbol

        try:
            # Get current price
            req = StockLatestQuoteRequest(symbol_or_symbols=alpaca_sym, feed=DataFeed.IEX)
            quote = data_client.get_stock_latest_quote(req)
            bid = float(quote[alpaca_sym].bid_price or 0)
            ask = float(quote[alpaca_sym].ask_price or 0)
            current_mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0

            if current_mid <= 0 or entry_price <= 0:
                continue

            # Calculate missed P&L
            if direction == "buy":
                missed_pct = (current_mid - entry_price) / entry_price
            else:
                missed_pct = (entry_price - current_mid) / entry_price

            missed_bps = round(missed_pct * 10000, 2)

            if abs(missed_bps) > 10:  # only log meaningful misses
                print(f"[TCA] MISSED: {symbol} {direction.upper()} skipped ({signal['skip_reason']}) → {missed_bps:+.0f}bps left on table")

            _log_tca_metric("opportunity_cost", {
                "symbol": symbol,
                "direction": direction,
                "skip_reason": signal["skip_reason"],
                "ensemble_score": signal["ensemble_score"],
                "entry_price": entry_price,
                "current_price": current_mid,
                "missed_alpha_bps": missed_bps,
                "hours_elapsed": round((datetime.now(timezone.utc) - signal["ts"]).total_seconds() / 3600, 2),
            })

        except Exception as e:
            print(f"[TCA] Opportunity cost error for {symbol}: {e}")


# ── Aggregate Stats ──────────────────────────────────────────────────────────

def get_adverse_selection_summary() -> dict:
    """Return per-symbol average adverse selection over rolling window."""
    summary = {}
    for symbol, records in _adverse_selection.items():
        if not records:
            continue
        recent = list(records)
        summary[symbol] = {
            "avg_1s_bps": round(np.mean([r["1s"] for r in recent]), 2),
            "avg_10s_bps": round(np.mean([r["10s"] for r in recent]), 2),
            "avg_60s_bps": round(np.mean([r["60s"] for r in recent]), 2),
            "n_fills": len(recent),
        }
    return summary


# ── Supabase Logging ─────────────────────────────────────────────────────────

def _log_tca_metric(metric_type: str, data: dict):
    """Write TCA metric to Supabase."""
    if not USER_ID:
        return
    try:
        supabase.table("tca_metrics").insert({
            "metric_type": metric_type,
            "data": data,
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        # TCA table might not exist yet — non-fatal
        if "relation" in str(e).lower() and "does not exist" in str(e).lower():
            pass  # table not created yet
        else:
            print(f"[TCA] Supabase log error: {e}")
