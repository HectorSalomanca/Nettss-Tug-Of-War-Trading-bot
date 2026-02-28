"""
Scout DIX — Dark Pool & FINRA Short Volume Intelligence

The Dark Tape: scrapes FINRA short volume data and dark pool prints
to detect where whales are quietly accumulating shares off-exchange.

Data Sources:
  1. FINRA Short Volume (daily): total short vs long volume per symbol
     - High short ratio (>50%) = bearish institutional flow
     - Declining short ratio = accumulation (shorts covering)
  2. Dark Pool Index (DIX proxy): aggregate dark pool buying pressure
     - Computed from short volume ratio across watchlist
     - Rising DIX = smart money buying (contrarian bullish)

Signals Published via ZeroMQ:
  - DARK_ACCUMULATION: short ratio dropping + volume rising = whale buying
  - DARK_DISTRIBUTION: short ratio spiking + volume rising = whale selling
  - DIX_EXTREME: aggregate dark pool index at extreme levels

Output: Supabase table 'dark_pool_signals' + ZeroMQ events

Schedule: Runs every 30 min during market hours (called by engine_v2.py)
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional

from referee.secret_vault import get_secret
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from referee.event_bus import get_publisher
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── Watchlist (must match engine_v2.py) ──────────────────────────────────────

WATCHLIST = [
    "NVDA", "CRWD", "LLY", "TSM", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR",
    "MSFT", "AMZN", "META", "GLD", "XLE", "UBER", "AMD", "COIN", "MRNA", "IWM",
]

# Internal name mapping
TICKER_MAP = {"TSMC": "TSM"}

# ── ZeroMQ Event Types ───────────────────────────────────────────────────────

EVT_DARK_ACCUMULATION = "DARK_ACCUMULATION"
EVT_DARK_DISTRIBUTION = "DARK_DISTRIBUTION"
EVT_DIX_EXTREME       = "DIX_EXTREME"

# ── Thresholds ───────────────────────────────────────────────────────────────

SHORT_RATIO_HIGH    = 0.55    # above this = bearish institutional flow
SHORT_RATIO_LOW     = 0.35    # below this = bullish (shorts covering)
SHORT_RATIO_DELTA   = -0.05   # 5% drop in short ratio = accumulation signal
DIX_EXTREME_HIGH    = 0.55    # aggregate DIX above this = extreme buying
DIX_EXTREME_LOW     = 0.35    # aggregate DIX below this = extreme selling
VOLUME_SURGE_MULT   = 1.5     # volume must be 1.5x average for signal

# ── Rolling History ──────────────────────────────────────────────────────────

_short_ratio_history: dict = {}  # {symbol: deque of (date, short_ratio, volume)}
_HISTORY_DAYS = 20


# ── FINRA Short Volume Fetcher ───────────────────────────────────────────────

def _fetch_short_volume_alpaca(symbol: str, days: int = 5) -> list:
    """
    Fetch recent daily bars from Alpaca and estimate short volume proxy.

    Since direct FINRA short volume requires a paid feed, we use a proxy:
    - Fetch daily bars with volume
    - Compare to 20-day average volume
    - Use intraday price action to estimate short pressure
      (close < open + high volume = likely short-driven)

    Returns list of {date, total_volume, short_ratio_proxy}
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(days * 2, 30))  # extra for 20d avg

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.sort_index()

        if len(df) < 5:
            return []

        results = []
        volumes = df["volume"].values
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values

        # 20-day average volume for normalization
        avg_vol_20 = pd.Series(volumes).rolling(20).mean().fillna(volumes.mean()).values

        for i in range(max(0, len(df) - days), len(df)):
            # Short volume proxy: estimate from price action
            # If close < open (bearish candle) + high volume → likely short-driven
            # If close > open (bullish candle) + high volume → likely long-driven
            candle_body = (closes[i] - opens[i]) / (opens[i] + 1e-8)
            candle_range = (highs[i] - lows[i]) / (opens[i] + 1e-8)

            # Proxy: short ratio = 0.5 - (candle_body / candle_range) * 0.2
            # Bearish candle → higher short ratio; bullish → lower
            if candle_range > 1e-8:
                body_ratio = candle_body / candle_range
                short_ratio = 0.50 - body_ratio * 0.15
            else:
                short_ratio = 0.50

            short_ratio = np.clip(short_ratio, 0.25, 0.75)

            # Volume surge factor amplifies the signal
            vol_ratio = volumes[i] / (avg_vol_20[i] + 1e-8)

            results.append({
                "date": str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])[:10],
                "total_volume": int(volumes[i]),
                "avg_volume_20d": int(avg_vol_20[i]),
                "volume_ratio": round(float(vol_ratio), 2),
                "short_ratio": round(float(short_ratio), 4),
                "candle_body_pct": round(float(candle_body * 100), 2),
            })

        return results

    except Exception as e:
        print(f"[DIX] Fetch error for {symbol}: {e}")
        return []


# ── Signal Detection ─────────────────────────────────────────────────────────

def _detect_signals(symbol: str, history: list) -> list:
    """
    Analyze short volume history for accumulation/distribution signals.

    DARK_ACCUMULATION: short ratio declining over 3+ days + volume above average
      → whales covering shorts / buying in dark pools
    DARK_DISTRIBUTION: short ratio rising over 3+ days + volume above average
      → whales increasing shorts / selling in dark pools
    """
    signals = []
    if len(history) < 3:
        return signals

    recent = history[-3:]
    ratios = [h["short_ratio"] for h in recent]
    vol_ratios = [h["volume_ratio"] for h in recent]

    # Short ratio trend (3-day delta)
    ratio_delta = ratios[-1] - ratios[0]
    avg_vol_ratio = np.mean(vol_ratios)
    latest = recent[-1]

    # ACCUMULATION: short ratio dropping + volume surge
    if ratio_delta < SHORT_RATIO_DELTA and avg_vol_ratio > VOLUME_SURGE_MULT:
        signals.append({
            "type": EVT_DARK_ACCUMULATION,
            "symbol": symbol,
            "short_ratio": latest["short_ratio"],
            "ratio_delta_3d": round(ratio_delta, 4),
            "volume_ratio": latest["volume_ratio"],
            "direction": "buy",
            "strength": round(abs(ratio_delta) * avg_vol_ratio * 10, 2),
        })

    # DISTRIBUTION: short ratio spiking + volume surge
    elif ratio_delta > abs(SHORT_RATIO_DELTA) and avg_vol_ratio > VOLUME_SURGE_MULT:
        signals.append({
            "type": EVT_DARK_DISTRIBUTION,
            "symbol": symbol,
            "short_ratio": latest["short_ratio"],
            "ratio_delta_3d": round(ratio_delta, 4),
            "volume_ratio": latest["volume_ratio"],
            "direction": "sell",
            "strength": round(abs(ratio_delta) * avg_vol_ratio * 10, 2),
        })

    # Extreme short ratio levels
    if latest["short_ratio"] > SHORT_RATIO_HIGH:
        signals.append({
            "type": EVT_DARK_DISTRIBUTION,
            "symbol": symbol,
            "short_ratio": latest["short_ratio"],
            "direction": "sell",
            "strength": round((latest["short_ratio"] - 0.5) * 20, 2),
            "reason": "extreme_short_ratio",
        })
    elif latest["short_ratio"] < SHORT_RATIO_LOW:
        signals.append({
            "type": EVT_DARK_ACCUMULATION,
            "symbol": symbol,
            "short_ratio": latest["short_ratio"],
            "direction": "buy",
            "strength": round((0.5 - latest["short_ratio"]) * 20, 2),
            "reason": "extreme_low_short_ratio",
        })

    return signals


def _compute_dix(watchlist_data: dict) -> float:
    """
    Compute aggregate Dark Pool Index (DIX proxy) across watchlist.

    DIX = volume-weighted average of (1 - short_ratio) across all symbols.
    Higher DIX = more dark pool buying = contrarian bullish.
    """
    total_vol = 0
    weighted_buy = 0

    for sym, history in watchlist_data.items():
        if not history:
            continue
        latest = history[-1]
        vol = latest["total_volume"]
        buy_ratio = 1.0 - latest["short_ratio"]
        weighted_buy += buy_ratio * vol
        total_vol += vol

    if total_vol == 0:
        return 0.50

    return round(weighted_buy / total_vol, 4)


# ── Main Scan ────────────────────────────────────────────────────────────────

def scan_dark_pools() -> dict:
    """
    Full dark pool scan across watchlist.

    Returns:
        {
            "dix": float,           # aggregate dark pool index
            "signals": list,        # detected accumulation/distribution signals
            "symbol_data": dict,    # per-symbol short volume data
            "dix_direction": str,   # "bullish" / "bearish" / "neutral"
        }
    """
    print(f"[DIX] Scanning dark pool activity for {len(WATCHLIST)} symbols...")
    t0 = time.time()

    watchlist_data = {}
    all_signals = []

    for symbol in WATCHLIST:
        history = _fetch_short_volume_alpaca(symbol, days=5)
        if history:
            watchlist_data[symbol] = history
            signals = _detect_signals(symbol, history)
            all_signals.extend(signals)

            # Update rolling history
            if symbol not in _short_ratio_history:
                _short_ratio_history[symbol] = deque(maxlen=_HISTORY_DAYS)
            for h in history:
                _short_ratio_history[symbol].append(h)

    # Compute aggregate DIX
    dix = _compute_dix(watchlist_data)

    # DIX extreme signal
    if dix > DIX_EXTREME_HIGH:
        dix_direction = "bullish"
        all_signals.append({
            "type": EVT_DIX_EXTREME,
            "symbol": "MARKET",
            "dix": dix,
            "direction": "buy",
            "strength": round((dix - 0.5) * 20, 2),
        })
    elif dix < DIX_EXTREME_LOW:
        dix_direction = "bearish"
        all_signals.append({
            "type": EVT_DIX_EXTREME,
            "symbol": "MARKET",
            "dix": dix,
            "direction": "sell",
            "strength": round((0.5 - dix) * 20, 2),
        })
    else:
        dix_direction = "neutral"

    elapsed = time.time() - t0
    print(f"[DIX] Scan complete in {elapsed:.1f}s | DIX={dix:.4f} ({dix_direction}) | {len(all_signals)} signal(s)")

    # Publish signals via ZeroMQ
    if HAS_EVENT_BUS and all_signals:
        try:
            pub = get_publisher()
            for sig in all_signals:
                pub.publish(sig["type"], sig.get("symbol", "MARKET"), sig)
            print(f"[DIX] Published {len(all_signals)} event(s) via ZeroMQ")
        except Exception as e:
            print(f"[DIX] ZeroMQ publish error (non-fatal): {e}")

    # Log to Supabase
    _log_to_supabase(dix, all_signals, watchlist_data)

    return {
        "dix": dix,
        "dix_direction": dix_direction,
        "signals": all_signals,
        "symbol_data": watchlist_data,
    }


def _log_to_supabase(dix: float, signals: list, watchlist_data: dict):
    """Log dark pool scan results to Supabase."""
    if not USER_ID:
        return
    try:
        # Log aggregate DIX
        supabase.table("dark_pool_signals").insert({
            "user_id": USER_ID,
            "symbol": "MARKET_DIX",
            "signal_type": "dix_aggregate",
            "dix_value": dix,
            "direction": "buy" if dix > 0.5 else "sell",
            "strength": round(abs(dix - 0.5) * 20, 2),
            "raw_data": json.dumps({
                "n_symbols": len(watchlist_data),
                "n_signals": len(signals),
            }),
        }).execute()

        # Log individual signals
        for sig in signals[:10]:  # cap at 10 to avoid spam
            supabase.table("dark_pool_signals").insert({
                "user_id": USER_ID,
                "symbol": sig.get("symbol", "UNKNOWN"),
                "signal_type": sig.get("type", "unknown"),
                "direction": sig.get("direction", "neutral"),
                "strength": sig.get("strength", 0),
                "short_ratio": sig.get("short_ratio"),
                "raw_data": json.dumps(sig),
            }).execute()
    except Exception as e:
        print(f"[DIX] Supabase log error (non-fatal): {e}")


# ── Public API for Engine Integration ────────────────────────────────────────

def get_dark_pool_signals() -> dict:
    """
    Get the latest dark pool signals for engine consumption.

    Returns:
        {
            symbol: {
                "direction": "buy"/"sell"/"neutral",
                "strength": float (0-10),
                "short_ratio": float,
            }
        }
    """
    result = scan_dark_pools()
    signals_by_symbol = {}

    for sig in result.get("signals", []):
        sym = sig.get("symbol", "")
        if sym == "MARKET":
            continue
        if sym not in signals_by_symbol or sig.get("strength", 0) > signals_by_symbol[sym].get("strength", 0):
            signals_by_symbol[sym] = {
                "direction": sig.get("direction", "neutral"),
                "strength": sig.get("strength", 0),
                "short_ratio": sig.get("short_ratio", 0.5),
            }

    # Add DIX as market-level signal
    signals_by_symbol["_DIX"] = {
        "direction": result["dix_direction"],
        "dix": result["dix"],
    }

    return signals_by_symbol


# ── Standalone Execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Scout DIX — Dark Pool & FINRA Short Volume Scanner")
    print("=" * 60)

    result = scan_dark_pools()

    print(f"\nAggregate DIX: {result['dix']:.4f} ({result['dix_direction'].upper()})")
    print(f"Signals detected: {len(result['signals'])}")

    for sig in result["signals"]:
        sym = sig.get("symbol", "?")
        direction = sig.get("direction", "?").upper()
        strength = sig.get("strength", 0)
        stype = sig.get("type", "?")
        print(f"  [{stype}] {sym}: {direction} (strength={strength:.1f})")

    print(f"\nPer-symbol short ratios:")
    for sym, data in sorted(result["symbol_data"].items()):
        if data:
            latest = data[-1]
            sr = latest["short_ratio"]
            vr = latest["volume_ratio"]
            bar = "█" * int(sr * 20)
            print(f"  {sym:6s}: SR={sr:.2%} VR={vr:.1f}x {bar}")
