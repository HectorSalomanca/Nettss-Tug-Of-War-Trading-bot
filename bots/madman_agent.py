"""
Madman Agent V2 — Retail/Emotional Bot with OFI Contrarian Fade

Signals:
  - RSI (overbought/oversold)
  - 15-min pump detection
  - Volume spike
  - OFI Z-score from scout_tape (Institutional Iceberg detection)
  - Contrarian Fade: RSI > 85 AND OFI Z < -1.5 → SELL (fade the crowd)
  - Reddit/StockTwits sentiment from Supabase signals
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee.secret_vault import get_secret
from quant.ticker_profiles import get_profile, compute_dynamic_pump_threshold, is_spread_too_wide, get_behavior

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

RSI_OVERBOUGHT        = 70
RSI_OVERSOLD          = 30
RSI_EXUBERANT         = 85    # extreme retail FOMO
PUMP_THRESHOLD_FLAT   = 0.05  # fallback if no profile data
VOLUME_SPIKE_MULT     = 2.0
OFI_FADE_THRESHOLD    = -1.5  # OFI Z below this = institutional absorption → fade retail


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def get_intraday_bars(symbol: str, days: int = 3) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start, end=end,
            limit=500,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[MADMAN] Intraday error for {symbol}: {e}")
        return None


def get_daily_bars(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[MADMAN] Daily error for {symbol}: {e}")
        return None


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def detect_pump_15min(df_1min: Optional[pd.DataFrame], symbol: str = "", df_daily: Optional[pd.DataFrame] = None) -> tuple:
    if df_1min is None or len(df_1min) < 15:
        return False, 0.0
    recent = df_1min.tail(15)
    change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

    # Dynamic pump threshold based on ticker personality + realized vol
    if df_daily is not None and len(df_daily) > 20 and symbol:
        threshold = compute_dynamic_pump_threshold(symbol, df_daily["close"].values)
    else:
        threshold = PUMP_THRESHOLD_FLAT

    return change >= threshold, round(change, 4)


def detect_volume_spike(df_daily: Optional[pd.DataFrame]) -> tuple:
    if df_daily is None or len(df_daily) < 10:
        return False, 1.0
    avg_vol    = df_daily["volume"].iloc[:-1].mean()
    latest_vol = df_daily["volume"].iloc[-1]
    ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    return ratio >= VOLUME_SPIKE_MULT, round(ratio, 2)


# ── OFI from Supabase ─────────────────────────────────────────────────────────

def get_ofi(symbol: str) -> dict:
    try:
        result = (
            supabase.table("ofi_snapshots")
            .select("ofi_z_score, iceberg_detected, created_at")
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            age = (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            if age < 300:
                return {
                    "ofi_z_score": row["ofi_z_score"],
                    "iceberg": row["iceberg_detected"],
                    "stacked_imbalance": False,
                    "trapped_exhaustion": False,
                    "spread_pct": 0.0,
                }
    except Exception:
        pass
    return {"ofi_z_score": 0.0, "iceberg": False, "stacked_imbalance": False, "trapped_exhaustion": False, "spread_pct": 0.0}


# ── Retail Sentiment from Supabase ────────────────────────────────────────────

def get_retail_sentiment(symbol: str) -> tuple:
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence")
            .eq("symbol", symbol)
            .eq("bot", "madman")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        rows = result.data
        if not rows:
            return "neutral", 0.5
        buy_conf  = [r["confidence"] for r in rows if r["direction"] == "buy"]
        sell_conf = [r["confidence"] for r in rows if r["direction"] == "sell"]
        avg_buy  = np.mean(buy_conf)  if buy_conf  else 0.0
        avg_sell = np.mean(sell_conf) if sell_conf else 0.0
        if avg_buy > avg_sell and avg_buy > 0.55:
            return "buy", float(avg_buy)
        elif avg_sell > avg_buy and avg_sell > 0.55:
            return "sell", float(avg_sell)
        return "neutral", 0.5
    except Exception:
        return "neutral", 0.5


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze(symbol: str, regime_state: str = "trend") -> dict:
    print(f"[MADMAN] Analyzing {symbol} (regime={regime_state})...")

    df_1min  = get_intraday_bars(symbol, days=3)
    df_daily = get_daily_bars(symbol, days=30)

    if df_daily is None or len(df_daily) < 5:
        return _neutral(symbol)

    closes_daily = df_daily["close"].values
    rsi = compute_rsi(closes_daily)

    is_pump, pump_pct   = detect_pump_15min(df_1min, symbol=symbol, df_daily=df_daily)
    is_vol_spike, vol_ratio = detect_volume_spike(df_daily)

    ofi_data    = get_ofi(symbol)
    ofi_z       = ofi_data["ofi_z_score"]
    is_iceberg  = ofi_data["iceberg"]
    is_stacked  = ofi_data.get("stacked_imbalance", False)
    is_trapped  = ofi_data.get("trapped_exhaustion", False)
    spread_pct  = ofi_data.get("spread_pct", 0.0)

    sentiment_dir, sentiment_conf = get_retail_sentiment(symbol)

    # Ticker personality
    profile  = get_profile(symbol)
    behavior = profile["behavior"]

    # Spread decay gate: if spread too wide, reduce confidence
    spread_gate = is_spread_too_wide(symbol, spread_pct)

    fomo  = 0
    fear  = 0
    details = {"rsi": rsi, "pump_pct": pump_pct, "vol_ratio": vol_ratio,
               "ofi_z_score": ofi_z, "iceberg_detected": is_iceberg,
               "stacked_imbalance": is_stacked, "trapped_exhaustion": is_trapped,
               "spread_pct": spread_pct, "spread_gate": spread_gate,
               "ticker_behavior": behavior}

    # ── CONTRARIAN FADE (highest priority signal) ─────────────
    # Retail exuberant (RSI > 85) but smart money absorbing (OFI Z < -1.5)
    # → Institutional iceberg is selling into retail buying → SELL
    if rsi >= RSI_EXUBERANT and ofi_z < OFI_FADE_THRESHOLD:
        fear += 5
        details["contrarian_fade"] = True
        print(f"[MADMAN] {symbol}: CONTRARIAN FADE triggered (RSI={rsi}, OFI Z={ofi_z:.2f})")
    else:
        details["contrarian_fade"] = False

    # ── Standard RSI ──────────────────────────────────────────
    if rsi >= RSI_OVERBOUGHT and not details["contrarian_fade"]:
        fomo += 2
        details["rsi_overbought"] = True
    elif rsi <= RSI_OVERSOLD:
        fear += 2
        details["rsi_oversold"] = True

    # ── Pump ──────────────────────────────────────────────────
    if is_pump:
        fomo += 3
        details["pump_detected"] = True
    else:
        details["pump_detected"] = False

    # ── Volume spike ──────────────────────────────────────────
    if is_vol_spike:
        fomo += 1
        details["volume_spike"] = True
    else:
        details["volume_spike"] = False

    # ── Iceberg (institutional absorption) ────────────────────
    if is_iceberg:
        fear += 2
        details["iceberg_signal"] = True
    else:
        details["iceberg_signal"] = False

    # ── Stacked Imbalance (V3: high-conviction retail momentum) ──
    if is_stacked:
        fomo += 3
        details["stacked_signal"] = True
    else:
        details["stacked_signal"] = False

    # ── Trapped Exhaustion (V3: retail trapped at peak) ────────
    if is_trapped:
        fear += 4  # strong mean-reversion signal
        details["trapped_signal"] = True
    else:
        details["trapped_signal"] = False

    # ── Retail sentiment ──────────────────────────────
    if sentiment_dir == "buy":
        fomo += 2
    elif sentiment_dir == "sell":
        fear += 2
    details["retail_sentiment"] = sentiment_dir

    # ── Behavior-adjusted scoring ─────────────────────────
    # Mean-reversion tickers: boost fear signals (fade retail)
    # Momentum tickers: boost fomo signals (ride retail)
    if behavior == "mean_reversion":
        fear = int(fear * 1.3)
    elif behavior == "momentum_breakout":
        fomo = int(fomo * 1.2)

    total = fomo + fear
    if total == 0:
        direction  = "neutral"
        confidence = 0.5
    elif fomo > fear:
        direction  = "buy"
        confidence = round(min(0.5 + (fomo / total) * 0.45, 0.95), 4)
    else:
        direction  = "sell"
        confidence = round(min(0.5 + (fear / total) * 0.45, 0.95), 4)

    # Spread decay gate: reduce confidence if spread is too wide
    if spread_gate:
        confidence = round(max(confidence * 0.7, 0.5), 4)
        details["spread_penalty"] = True
    else:
        details["spread_penalty"] = False

    raw_data = {
        **details,
        "current_price": float(closes_daily[-1]),
        "fomo_score": fomo,
        "fear_score": fear,
        "sentiment_conf": sentiment_conf,
        "regime": regime_state,
    }

    print(f"[MADMAN] {symbol}: {direction.upper()} @ {confidence:.2%} | RSI={rsi} | OFI Z={ofi_z:.2f} | Fade={details['contrarian_fade']}")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "signal_type": "rsi_ofi_contrarian",
        "raw_data": raw_data,
    }


def _neutral(symbol: str) -> dict:
    return {"symbol": symbol, "direction": "neutral", "confidence": 0.5,
            "signal_type": "insufficient_data", "raw_data": {}}


def log_signal(result: dict, regime_state: str = "trend"):
    if not USER_ID:
        return
    record = {
        "symbol": result["symbol"],
        "bot": "madman",
        "direction": result["direction"],
        "confidence": result["confidence"],
        "signal_type": result["signal_type"],
        "regime_state": regime_state,
        "ofi_z_score": result["raw_data"].get("ofi_z_score"),
        "raw_data": result["raw_data"],
        "user_id": USER_ID,
    }
    try:
        supabase.table("signals").insert(record).execute()
    except Exception as e:
        print(f"[MADMAN] Supabase error: {e}")


def run(symbols: list, regime_state: str = "trend") -> list:
    results = []
    for symbol in symbols:
        result = analyze(symbol, regime_state)
        log_signal(result, regime_state)
        results.append(result)
    return results


if __name__ == "__main__":
    test = ["NVDA", "PLTR", "JPM"]
    for r in run(test):
        print(r)
