"""
Sovereign Agent V2 — Institutional/Rational Bot

Signals:
  - Adaptive Fractional Differencing (AFD) momentum (from feature_factory)
  - Pure Alpha (SPY/QQQ-neutralized returns)
  - Opening Range Breakout (ORB): first 30-min high/low sets the range
  - SEC 8-K / news sentiment (from scout_news Supabase signals)
  - Kelly Criterion position sizing

Holds until 3:55 PM ET or stop/take fires.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.feature_factory import build_features
from quant.stockformer import predict as stockformer_predict
from scout.scout_alt import get_alt_signal

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

ET = pytz.timezone("America/New_York")
ORB_MINUTES = 30   # Opening Range = first 30 min


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def get_daily_bars(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
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
        print(f"[SOVEREIGN] Daily bars error for {symbol}: {e}")
        return None


def get_benchmark_bars(symbol: str, days: int = 90) -> Optional[np.ndarray]:
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
        return df.sort_index()["close"].values
    except Exception:
        return None


def get_intraday_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch today's 1-min bars for ORB calculation."""
    try:
        now_et = datetime.now(ET)
        start_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        start_utc = start_et.astimezone(timezone.utc)
        end_utc = datetime.now(timezone.utc)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_utc, end=end_utc,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index()
    except Exception:
        return None


# ── Opening Range Breakout ────────────────────────────────────────────────────

def compute_orb(df_1min: Optional[pd.DataFrame]) -> dict:
    """
    Returns ORB signal: direction + strength based on current price vs range.
    """
    if df_1min is None or len(df_1min) < ORB_MINUTES:
        return {"orb_direction": "neutral", "orb_strength": 0.0, "orb_range": 0.0}

    orb_bars = df_1min.iloc[:ORB_MINUTES]
    orb_high = float(orb_bars["high"].max())
    orb_low  = float(orb_bars["low"].min())
    orb_range = orb_high - orb_low

    current_price = float(df_1min["close"].iloc[-1])

    if orb_range == 0:
        return {"orb_direction": "neutral", "orb_strength": 0.0, "orb_range": 0.0}

    if current_price > orb_high:
        strength = min((current_price - orb_high) / orb_range, 1.0)
        return {"orb_direction": "buy", "orb_strength": round(strength, 4), "orb_range": round(orb_range, 4)}
    elif current_price < orb_low:
        strength = min((orb_low - current_price) / orb_range, 1.0)
        return {"orb_direction": "sell", "orb_strength": round(strength, 4), "orb_range": round(orb_range, 4)}

    return {"orb_direction": "neutral", "orb_strength": 0.0, "orb_range": round(orb_range, 4)}


# ── News Sentiment from Supabase ──────────────────────────────────────────────

def get_news_sentiment(symbol: str) -> tuple:
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence, bayesian_score, created_at")
            .eq("symbol", symbol)
            .eq("signal_type", "8k_news_bayesian")
            .order("created_at", desc=True)
            .limit(3)
            .execute()
        )
        rows = result.data
        if not rows:
            return "neutral", 0.5, 0.5

        buy_conf  = [r["confidence"] for r in rows if r["direction"] == "buy"]
        sell_conf = [r["confidence"] for r in rows if r["direction"] == "sell"]
        avg_bayes = np.mean([r.get("bayesian_score", 0.5) for r in rows])

        avg_buy  = np.mean(buy_conf)  if buy_conf  else 0.0
        avg_sell = np.mean(sell_conf) if sell_conf else 0.0

        if avg_buy > avg_sell and avg_buy > 0.55:
            return "buy", float(avg_buy), float(avg_bayes)
        elif avg_sell > avg_buy and avg_sell > 0.55:
            return "sell", float(avg_sell), float(avg_bayes)
        return "neutral", 0.5, float(avg_bayes)
    except Exception:
        return "neutral", 0.5, 0.5


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def compute_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.005  # floor at 0.5%
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    return max(0.005, min(kelly * 0.5, 0.25))  # floor 0.5%, cap 25%


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze(symbol: str, regime_state: str = "trend") -> dict:
    print(f"[SOVEREIGN] Analyzing {symbol} (regime={regime_state})...")

    df = get_daily_bars(symbol, days=90)
    if df is None or len(df) < 20:
        return _neutral(symbol)

    prices = df["close"].values
    spy_prices = get_benchmark_bars("SPY", days=90)
    qqq_prices = get_benchmark_bars("QQQ", days=90)

    # ── Feature Factory ───────────────────────────────────────
    if spy_prices is not None and qqq_prices is not None:
        features = build_features(prices, spy_prices, qqq_prices)
    else:
        features = {"afd_momentum": 0.0, "pure_alpha": 0.0, "d_value": 1.0, "neutralized_vol": 0.3}

    afd_mom   = features["afd_momentum"]
    pure_alpha = features["pure_alpha"]
    neut_vol  = features["neutralized_vol"]

    # ── ORB ───────────────────────────────────────────────────
    df_1min = get_intraday_bars(symbol)
    orb = compute_orb(df_1min)

    # ── News Sentiment ────────────────────────────────────────
    news_dir, news_conf, bayesian_score = get_news_sentiment(symbol)

    # ── Kelly (P6: use 5-day returns to match holding period) ──
    if len(prices) >= 6:
        returns_5d = (prices[5:] - prices[:-5]) / prices[:-5]
    else:
        returns_5d = np.diff(prices) / prices[:-1]
    wins    = returns_5d[returns_5d > 0]
    losses  = returns_5d[returns_5d < 0]
    win_rate  = len(wins) / len(returns_5d) if len(returns_5d) > 0 else 0.5
    avg_win   = float(np.mean(wins))   if len(wins)   > 0 else 0.01
    avg_loss  = float(abs(np.mean(losses))) if len(losses) > 0 else 0.01
    kelly_fraction = compute_kelly(win_rate, avg_win, avg_loss)

    # ── Stockformer Conviction (V3) ─────────────────────────
    sf_scores = stockformer_predict({})
    sf_score = sf_scores.get(symbol, 0.0)

    # ── Alt Data Signal (V3) ────────────────────────────────
    alt_signal = get_alt_signal(symbol)
    alt_dir = alt_signal["direction"]
    alt_conf = alt_signal["confidence"]

    # ── RSI + MACD + Volume Surge + 52W Position + MA20 ──────────────
    rsi_val = 50.0
    macd_signal = 0.0
    vol_surge_ratio = 1.0   # today vol / 20d avg vol
    pct_52w = 0.5           # 0=52w low, 1=52w high
    price_vs_ma20 = 0.0     # % above/below 20-day MA
    try:
        close_s = pd.Series(prices)
        vol_s   = pd.Series(df["volume"].values)

        # RSI
        delta = close_s.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi_val = float((100 - 100 / (1 + rs)).iloc[-1])

        # MACD histogram
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        macd_signal = float(macd_line.iloc[-1] - macd_sig.iloc[-1])

        # Volume surge: today's vol vs 20-day average
        avg_vol_20 = float(vol_s.rolling(20).mean().iloc[-1])
        today_vol  = float(vol_s.iloc[-1])
        vol_surge_ratio = today_vol / max(avg_vol_20, 1)

        # 52-week position: where is price in its annual range?
        hi_52 = float(close_s.rolling(min(252, len(close_s))).max().iloc[-1])
        lo_52 = float(close_s.rolling(min(252, len(close_s))).min().iloc[-1])
        rng = hi_52 - lo_52
        pct_52w = (prices[-1] - lo_52) / rng if rng > 0 else 0.5

        # Price vs 20-day MA
        ma20 = float(close_s.rolling(20).mean().iloc[-1])
        price_vs_ma20 = (prices[-1] - ma20) / ma20
    except Exception:
        pass

    # ── Score Aggregation ─────────────────────────────────────
    bull = 0
    bear = 0
    details = {}

    # AFD momentum
    if afd_mom > 0.0001:
        bull += 2
        details["afd_bullish"] = True
    elif afd_mom < -0.0001:
        bear += 2
        details["afd_bullish"] = False
    details["afd_momentum"] = round(afd_mom, 6)

    # Pure alpha
    if pure_alpha > 0.0005:
        bull += 2
        details["pure_alpha_bullish"] = True
    elif pure_alpha < -0.0005:
        bear += 2
        details["pure_alpha_bullish"] = False
    details["pure_alpha"] = round(pure_alpha, 6)

    # ORB
    if orb["orb_direction"] == "buy":
        bull += 3
        details["orb_breakout"] = "up"
    elif orb["orb_direction"] == "sell":
        bear += 3
        details["orb_breakout"] = "down"
    else:
        details["orb_breakout"] = "none"
    details["orb_strength"] = orb["orb_strength"]

    # News
    if news_dir == "buy":
        bull += 2
        details["news_bullish"] = True
    elif news_dir == "sell":
        bear += 2
        details["news_bullish"] = False
    details["news_direction"] = news_dir
    details["bayesian_score"] = round(bayesian_score, 4)

    # Stockformer conviction (V3: highest weight — 4 points)
    if sf_score > 0.15:
        bull += 4
        details["stockformer_bullish"] = True
    elif sf_score < -0.15:
        bear += 4
        details["stockformer_bullish"] = False
    details["stockformer_score"] = round(sf_score, 4)

    # Alt data signal (V3: 2 points, high-alpha sources)
    if alt_dir == "buy" and alt_conf > 0.6:
        bull += 2
        details["alt_data_bullish"] = True
    elif alt_dir == "sell" and alt_conf > 0.6:
        bear += 2
        details["alt_data_bullish"] = False
    details["alt_data_direction"] = alt_dir
    details["alt_data_confidence"] = round(alt_conf, 4)

    # RSI (2 points): oversold <35 = bullish, overbought >65 = bearish
    if rsi_val < 35:
        bull += 2
        details["rsi_signal"] = "oversold"
    elif rsi_val > 65:
        bear += 2
        details["rsi_signal"] = "overbought"
    else:
        details["rsi_signal"] = "neutral"
    details["rsi"] = round(rsi_val, 2)

    # MACD histogram (2 points): positive = bullish momentum, negative = bearish
    if macd_signal > 0:
        bull += 2
        details["macd_signal"] = "bullish"
    elif macd_signal < 0:
        bear += 2
        details["macd_signal"] = "bearish"
    else:
        details["macd_signal"] = "neutral"
    details["macd_histogram"] = round(macd_signal, 4)

    # Volume surge (2 points): >1.5x avg vol = conviction; <0.6x = fading
    if vol_surge_ratio > 1.5:
        # Volume surge confirms direction of price move
        if price_vs_ma20 > 0:
            bull += 2
        else:
            bear += 2
        details["vol_surge"] = "high"
    elif vol_surge_ratio < 0.6:
        details["vol_surge"] = "low"
    else:
        details["vol_surge"] = "normal"
    details["vol_surge_ratio"] = round(vol_surge_ratio, 2)

    # 52-week position (2 points): near highs = momentum, near lows = mean-reversion
    if pct_52w > 0.80:
        bull += 2   # near 52w high = momentum breakout
        details["52w_position"] = "near_high"
    elif pct_52w < 0.20:
        bull += 2   # near 52w low = mean-reversion bounce candidate
        details["52w_position"] = "near_low"
    else:
        details["52w_position"] = "mid_range"
    details["pct_52w"] = round(pct_52w, 3)

    # Price vs 20-day MA (2 points): trend direction filter
    if price_vs_ma20 > 0.01:   # >1% above MA20 = uptrend
        bull += 2
        details["ma20_signal"] = "above"
    elif price_vs_ma20 < -0.01:  # >1% below MA20 = downtrend
        bear += 2
        details["ma20_signal"] = "below"
    else:
        details["ma20_signal"] = "at_ma"
    details["price_vs_ma20"] = round(price_vs_ma20, 4)

    total = bull + bear
    if total == 0:
        direction = "neutral"
        base_conf = 0.5
    elif bull > bear:
        direction = "buy"
        base_conf = bull / total
    else:
        direction = "sell"
        base_conf = bear / total

    # Volatility penalty: high idiosyncratic vol reduces confidence
    # (but does NOT compress toward 50% — that was killing all signals)
    vol_penalty = min(neut_vol / 3.0, 0.10)
    confidence = round(max(0.5, min(base_conf - vol_penalty, 0.95)), 4)

    raw_data = {
        **details,
        "current_price": float(prices[-1]),
        "kelly_fraction": round(kelly_fraction, 4),
        "win_rate": round(win_rate, 4),
        "neutralized_vol": round(neut_vol, 4),
        "d_value": features["d_value"],
        "bull_factors": bull,
        "bear_factors": bear,
        "regime": regime_state,
    }

    print(f"[SOVEREIGN] {symbol}: {direction.upper()} @ {confidence:.2%} | AFD={afd_mom:.5f} | ORB={orb['orb_direction']} | Kelly={kelly_fraction:.2%}")

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "signal_type": "afd_orb_sovereign",
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
        "bot": "sovereign",
        "direction": result["direction"],
        "confidence": result["confidence"],
        "signal_type": result["signal_type"],
        "regime_state": regime_state,
        "afd_momentum": result["raw_data"].get("afd_momentum"),
        "raw_data": result["raw_data"],
        "user_id": USER_ID,
    }
    try:
        supabase.table("signals").insert(record).execute()
    except Exception as e:
        print(f"[SOVEREIGN] Supabase error: {e}")


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
