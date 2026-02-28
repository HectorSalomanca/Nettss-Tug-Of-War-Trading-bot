"""
Triple Barrier Labeler V3 — Institutional-Grade Trade Labeling

Labels historical trades based on which barrier is hit first:
  - Upper Barrier (+4%): Take Profit → label = 1 (win)
  - Lower Barrier (-2%): Stop Loss   → label = -1 (loss)
  - Vertical Barrier (3:45 PM ET):   → label = 0 (time stop)

Runs nightly on all trades in Supabase that lack a tbl_label.
Labels are used as training targets for Stockformer weekly retraining.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytz
from dateutil import parser as dateutil_parser
from referee.secret_vault import get_secret
from supabase import create_client, Client
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

ET = pytz.timezone("America/New_York")

# Barrier thresholds (fallback — dynamic ATR barriers override these)
UPPER_BARRIER_PCT = 0.04   # +4% take profit (fallback)
LOWER_BARRIER_PCT = 0.02   # -2% stop loss (fallback)
VERTICAL_HOUR     = 15     # 3:45 PM ET
VERTICAL_MINUTE   = 45

# ATR-based dynamic barrier multipliers (Lopez de Prado style)
ATR_TAKE_MULT = 1.5   # take profit = entry ± 1.5 * daily ATR
ATR_STOP_MULT = 0.75  # stop loss   = entry ± 0.75 * daily ATR (preserves 2:1 R/R)


def compute_atr_barriers(symbol: str, entry_time: datetime) -> dict:
    """
    Compute volatility-adjusted Triple Barrier thresholds using 20-day ATR.
    Returns {'upper_pct': float, 'lower_pct': float, 'atr': float}.
    Falls back to fixed barriers if data unavailable.
    """
    try:
        end = entry_time.astimezone(timezone.utc)
        start = end - timedelta(days=40)  # fetch extra for ATR warmup
        alpaca_sym = "TSM" if symbol == "TSMC" else symbol
        req = StockBarsRequest(
            symbol_or_symbols=alpaca_sym,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(alpaca_sym, level=0)
        df = df.sort_index()

        if len(df) < 15:
            return {"upper_pct": UPPER_BARRIER_PCT, "lower_pct": LOWER_BARRIER_PCT, "atr": 0.0}

        # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
        high = df["high"].values
        low  = df["low"].values
        close_prev = np.roll(df["close"].values, 1)
        close_prev[0] = df["close"].values[0]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - close_prev), np.abs(low - close_prev))
        )
        atr_20 = float(np.mean(tr[-20:]))  # 20-day ATR in dollar terms
        last_close = float(df["close"].iloc[-1])

        if last_close <= 0 or atr_20 <= 0:
            return {"upper_pct": UPPER_BARRIER_PCT, "lower_pct": LOWER_BARRIER_PCT, "atr": 0.0}

        # Convert ATR to percentage of price
        atr_pct = atr_20 / last_close
        upper = round(atr_pct * ATR_TAKE_MULT, 6)
        lower = round(atr_pct * ATR_STOP_MULT, 6)

        # Sanity clamps: never wider than 8% or tighter than 0.5%
        upper = max(0.005, min(upper, 0.08))
        lower = max(0.003, min(lower, 0.04))

        return {"upper_pct": upper, "lower_pct": lower, "atr": round(atr_pct, 6)}

    except Exception as e:
        print(f"[LABELER] ATR compute error for {symbol}: {e}")
        return {"upper_pct": UPPER_BARRIER_PCT, "lower_pct": LOWER_BARRIER_PCT, "atr": 0.0}


def get_intraday_bars(symbol: str, date: datetime) -> Optional[pd.DataFrame]:
    """Fetch 1-min bars for a specific trading day."""
    try:
        start = date.replace(hour=9, minute=30, second=0, microsecond=0, tzinfo=ET)
        end   = date.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=ET)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[LABELER] Bar fetch error for {symbol} on {date.date()}: {e}")
        return None


def label_trade(
    symbol: str,
    entry_price: float,
    entry_time: datetime,
    side: str,
) -> dict:
    """
    Apply Triple Barrier Labeling with ATR-based dynamic barriers.

    Returns:
      - tbl_label: 1 (win), -1 (loss), 0 (time stop)
      - barrier_hit: "upper", "lower", "vertical"
      - exit_price: price at barrier hit
      - exit_time: timestamp of barrier hit
      - max_favorable: maximum favorable excursion (%)
      - max_adverse: maximum adverse excursion (%)
      - sample_weight: higher for fast wins, lower for slow time-stops
      - upper_barrier_pct / lower_barrier_pct: actual thresholds used
    """
    trade_date = entry_time.astimezone(ET)
    df = get_intraday_bars(symbol, trade_date)

    if df is None or len(df) < 10:
        return {"tbl_label": 0, "barrier_hit": "no_data", "exit_price": entry_price, "sample_weight": 0.1}

    # Filter to bars after entry
    entry_utc = entry_time.astimezone(timezone.utc) if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
    df_after = df[df.index >= entry_utc]
    if len(df_after) == 0:
        return {"tbl_label": 0, "barrier_hit": "no_bars_after_entry", "exit_price": entry_price, "sample_weight": 0.1}

    # Dynamic ATR barriers (per-symbol, per-trade)
    barriers = compute_atr_barriers(symbol, entry_time)
    upper_pct = barriers["upper_pct"]
    lower_pct = barriers["lower_pct"]
    atr_pct   = barriers["atr"]

    # Vertical barrier: 3:45 PM ET
    vertical_time = trade_date.replace(hour=VERTICAL_HOUR, minute=VERTICAL_MINUTE, second=0)
    vertical_utc = vertical_time.astimezone(timezone.utc)

    max_favorable = 0.0
    max_adverse = 0.0
    total_bars = len(df_after)

    for bar_idx, (idx, row) in enumerate(df_after.iterrows()):
        current_price = float(row["close"])
        bar_time = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        if side == "buy":
            pct_change = (current_price - entry_price) / entry_price
        else:
            pct_change = (entry_price - current_price) / entry_price

        max_favorable = max(max_favorable, pct_change)
        max_adverse = min(max_adverse, pct_change)

        bars_elapsed = bar_idx + 1

        # Upper barrier (take profit) — ATR-scaled
        if pct_change >= upper_pct:
            # Sample weight: fast wins weighted higher (inverse of time to exit)
            speed_factor = max(0.2, 1.0 - (bars_elapsed / max(total_bars, 1)))
            magnitude_factor = min(abs(pct_change) / 0.02, 3.0)  # scale by return magnitude
            weight = round(speed_factor * magnitude_factor, 4)
            return {
                "tbl_label": 1,
                "barrier_hit": "upper",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": bars_elapsed,
                "sample_weight": weight,
                "upper_barrier_pct": upper_pct,
                "lower_barrier_pct": lower_pct,
                "atr_pct": atr_pct,
            }

        # Lower barrier (stop loss) — ATR-scaled
        if pct_change <= -lower_pct:
            speed_factor = max(0.2, 1.0 - (bars_elapsed / max(total_bars, 1)))
            magnitude_factor = min(abs(pct_change) / 0.02, 3.0)
            weight = round(speed_factor * magnitude_factor, 4)
            return {
                "tbl_label": -1,
                "barrier_hit": "lower",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": bars_elapsed,
                "sample_weight": weight,
                "upper_barrier_pct": upper_pct,
                "lower_barrier_pct": lower_pct,
                "atr_pct": atr_pct,
            }

        # Vertical barrier (time stop)
        if hasattr(bar_time, 'astimezone'):
            bar_et = bar_time.astimezone(ET)
        else:
            bar_et = bar_time
        if hasattr(bar_et, 'hour') and (bar_et.hour > VERTICAL_HOUR or
            (bar_et.hour == VERTICAL_HOUR and bar_et.minute >= VERTICAL_MINUTE)):
            # Time stops get low weight — they teach the model nothing
            return {
                "tbl_label": 0,
                "barrier_hit": "vertical",
                "exit_price": round(current_price, 4),
                "exit_time": str(bar_time),
                "max_favorable": round(max_favorable, 6),
                "max_adverse": round(max_adverse, 6),
                "bars_to_exit": bars_elapsed,
                "sample_weight": 0.25,
                "upper_barrier_pct": upper_pct,
                "lower_barrier_pct": lower_pct,
                "atr_pct": atr_pct,
            }

    # End of data without hitting any barrier
    last_price = float(df_after["close"].iloc[-1])
    return {
        "tbl_label": 0,
        "barrier_hit": "eod",
        "exit_price": round(last_price, 4),
        "max_favorable": round(max_favorable, 6),
        "max_adverse": round(max_adverse, 6),
        "sample_weight": 0.2,
        "upper_barrier_pct": upper_pct,
        "lower_barrier_pct": lower_pct,
        "atr_pct": atr_pct,
    }


def run_nightly_labeling():
    """
    Fetch all unlabeled trades from Supabase and apply Triple Barrier Labeling.
    Updates each trade with tbl_label.
    """
    print(f"[LABELER] Starting nightly labeling at {datetime.now(timezone.utc).isoformat()}")

    try:
        result = (
            supabase.table("trades")
            .select("id, symbol, side, limit_price, created_at, status")
            .is_("tbl_label", "null")
            .eq("status", "filled")
            .order("created_at", desc=False)
            .limit(100)
            .execute()
        )
    except Exception as e:
        print(f"[LABELER] Fetch error: {e}")
        return

    trades = result.data if result.data else []
    if not trades:
        print("[LABELER] No unlabeled trades found")
        return

    print(f"[LABELER] Found {len(trades)} unlabeled trades")

    labeled = 0
    for trade in trades:
        entry_time = dateutil_parser.parse(trade["created_at"])
        raw_price = trade.get("limit_price") or trade.get("fill_price")
        if raw_price is None:
            continue
        entry_price = float(raw_price)
        if entry_price <= 0:
            continue

        label_result = label_trade(
            symbol=trade["symbol"],
            entry_price=entry_price,
            entry_time=entry_time,
            side=trade["side"],
        )

        label_str = {1: "WIN", -1: "LOSS", 0: "TIME_STOP"}.get(label_result["tbl_label"], "UNKNOWN")
        atr_info = f" ATR={label_result.get('atr_pct', 0):.3%}" if label_result.get('atr_pct') else ""
        print(f"[LABELER] {trade['symbol']}: {label_str} via {label_result['barrier_hit']} "
              f"(MFE={label_result.get('max_favorable', 0):.2%}, MAE={label_result.get('max_adverse', 0):.2%}, "
              f"w={label_result.get('sample_weight', 0):.2f}{atr_info})")

        try:
            supabase.table("trades").update({
                "tbl_label": label_result["tbl_label"],
            }).eq("id", trade["id"]).execute()
            labeled += 1
        except Exception as e:
            print(f"[LABELER] Update error for {trade['id']}: {e}")

    print(f"[LABELER] Labeled {labeled}/{len(trades)} trades")


if __name__ == "__main__":
    run_nightly_labeling()
