"""
Tug-Of-War Referee Engine V2

Upgrades over V1:
  - New watchlist: NVDA, CRWD, LLY, TSMC, JPM, NEE, CAT, SONY, PLTR, MSTR
  - SPY/QQQ kept as silent benchmarks (not traded)
  - HMM regime gating: Crisis → halt all, Chop → tighter stops
  - Uses sovereign_agent + madman_agent (V2 bots)
  - Limit IOC orders instead of market orders (reduces slippage)
  - Logs implementation shortfall in basis points
  - Scout V2 (scout_news.py) runs every 30 min in background
  - HMM re-fits every 60 min in background
"""

import os
import sys
import json
import time
import schedule
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional
import pytz

from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, TakeProfitRequest, StopLossRequest,
    GetOrdersRequest, StopLimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bots import sovereign_agent, madman_agent
from referee import position_manager
from quant import regime_hmm
from quant.earnings_filter import filter_watchlist
from quant.correlation_guard import filter_by_correlation
from quant.regime_hmm import get_latest_regime_full
from quant.meta_model import compute_ensemble_score
from quant.stockformer import predict as stockformer_predict, SEQ_LEN, N_FEATURES
from quant.feature_factory import build_features
from referee.net_guard import is_online, with_retry

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")
PAPER_TRADE       = os.getenv("ALPACA_PAPER_TRADE", "True").lower() == "true"

supabase: Client  = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
trading_client    = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADE)
data_client       = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ── Watchlist ─────────────────────────────────────────────────────────────────
WATCHLIST   = ["NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR"]
BENCHMARKS  = ["SPY", "QQQ"]   # used for neutralization, not traded
CRISIS_ETF  = "SQQQ"           # 3× inverse QQQ — bought during Crisis

# Alpaca ticker mapping (some symbols differ from our watchlist names)
ALPACA_TICKER = {"TSMC": "TSM"}  # Taiwan Semiconductor ADR
CRISIS_ETF_RISK_PCT  = 0.03    # 3% equity into SQQQ hedge
CRISIS_SHORT_RISK_PCT = 0.02   # 2% equity into weakest-symbol short
CRISIS_CONF_THRESHOLD = 0.90   # only hedge if HMM is ≥90% confident in Crisis

# ── Position sizing ───────────────────────────────────────────────────────────
BASE_RISK_PCT   = 0.03    # 3% equity per trade (floor) — 2% was too small
MAX_RISK_PCT    = 0.06    # 6% equity hard cap
MAX_OPEN_TRADES = 4

# ── Execution thresholds ──────────────────────────────────────────────────────
SOVEREIGN_MIN_CONF  = 0.60   # lowered: old 75% was unreachable with neutralization
SOVEREIGN_SOLO_CONF = 0.70   # lowered: ensemble override provides real conviction
LIMIT_SLIP_PCT      = 0.0015   # limit price = mid ± 0.15% (wider — 0.05% was getting 100% IOC cancels)

# ── Regime-adjusted stops ─────────────────────────────────────────────────────
# Off-round numbers: trigger slightly before institutional round-number clusters
# (e.g. 1.9% stop fires before the crowd's 2% stops get swept)
REGIME_STOPS = {
    "trend":       {"stop": 0.019, "take": 0.038},
    "trend_bull":  {"stop": 0.024, "take": 0.048},
    "trend_bear":  {"stop": 0.014, "take": 0.029},
    "chop":        {"stop": 0.009, "take": 0.018},
    "crisis":      {"stop": 0.00,  "take": 0.00},
}
ENSEMBLE_BUY_THRESHOLD  = 0.12   # ensemble score > this → execute buy
ENSEMBLE_SELL_THRESHOLD = -0.12  # ensemble score < this → execute sell

ET = pytz.timezone("America/New_York")

# ── Crisis streak counter (Fix 1: require 2 consecutive crisis readings) ───────
_crisis_streak = 0   # incremented each cycle crisis is detected, reset otherwise
CRISIS_STREAK_REQUIRED = 2  # must see crisis N times in a row before acting

# ── Offline cache path ─────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR   = os.path.join(_BASE_DIR, "cache")
CACHE_FILE  = os.path.join(CACHE_DIR, "last_known.json")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _save_cache(regime: str, confidence: float, equity: float):
    """Persist last-known-good state to disk for offline cycles."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "regime": regime,
                "confidence": confidence,
                "equity": equity,
                "ts": datetime.now(timezone.utc).isoformat(),
            }, f)
    except Exception:
        pass


def _load_cache() -> dict:
    """Load last-known-good state. Returns defaults if file missing."""
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        age_mins = (datetime.now(timezone.utc) - datetime.fromisoformat(data["ts"])).seconds // 60
        print(f"[CACHE] Using last-known regime={data['regime'].upper()} from {age_mins} min ago")
        return data
    except Exception:
        return {"regime": "trend", "confidence": 0.5, "equity": 100000.0, "ts": None}


# ── Pending order reconciliation ───────────────────────────────────────────────

def _reconcile_pending_orders():
    """
    At cycle start: query Alpaca for all orders in last 24h and sync
    any stale 'pending' Supabase records to their actual fill status.
    Fixes the problem where DAY orders fill while laptop is offline.
    """
    if not USER_ID:
        return
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50)
        orders = trading_client.get_orders(filter=req)
        updated = 0
        for o in orders:
            oid = str(o.id)
            status_str = str(o.status).lower()
            filled_qty = float(o.filled_qty or 0)
            filled_price = float(o.filled_avg_price or 0)

            if "filled" in status_str and filled_qty > 0:
                r = supabase.table("trades").update({
                    "status": "filled",
                    "qty": int(filled_qty),
                    "limit_price": round(filled_price, 4),
                }).eq("alpaca_order_id", oid).eq("status", "pending").execute()
                if r.data:
                    updated += 1
                    print(f"[RECONCILE] {o.symbol}: filled {int(filled_qty)} @ ${filled_price:.2f} — Supabase updated")
            elif any(s in status_str for s in ("cancel", "expired", "rejected")):
                r = supabase.table("trades").update({
                    "status": "cancelled",
                }).eq("alpaca_order_id", oid).eq("status", "pending").execute()
                if r.data:
                    updated += 1
                    print(f"[RECONCILE] {o.symbol}: {status_str} — Supabase updated")
        if updated:
            print(f"[RECONCILE] Synced {updated} stale order(s)")
    except Exception as e:
        print(f"[RECONCILE] Error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_account_equity() -> float:
    try:
        return float(trading_client.get_account().equity)
    except Exception as e:
        print(f"[REFEREE] Account fetch error: {e}")
        return 100000.0


def is_market_open() -> bool:
    now_et  = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    open_  = now_et.replace(hour=4,  minute=0,  second=0, microsecond=0)
    close_ = now_et.replace(hour=20, minute=0,  second=0, microsecond=0)
    return open_ <= now_et <= close_


def is_high_noise_window() -> bool:
    """
    Return True during high-noise windows where new entries should be skipped:
      - First 30 min after open (9:30-10:00 ET): wide spreads, erratic price action
      - Last 15 min before close (3:45-4:00 ET): EOD positioning noise
    Exits (stop/take) still run — only new entries are blocked.
    """
    now_et = datetime.now(ET)
    h, m = now_et.hour, now_et.minute
    # 9:30-10:00 ET
    if h == 9 and m >= 30:
        return True
    if h == 10 and m < 0:  # exactly 10:00 is fine
        return True
    # 15:45-16:00 ET
    if h == 15 and m >= 45:
        return True
    return False


def get_open_position_count() -> int:
    try:
        return len(trading_client.get_all_positions())
    except Exception:
        return 0


def _alpaca_sym(symbol: str) -> str:
    """Map internal watchlist name to Alpaca ticker."""
    return ALPACA_TICKER.get(symbol, symbol)


def get_mid_price(symbol: str) -> float:
    asym = _alpaca_sym(symbol)
    try:
        req   = StockLatestQuoteRequest(symbol_or_symbols=asym, feed=DataFeed.IEX)
        quote = data_client.get_stock_latest_quote(req)
        bid   = float(quote[asym].bid_price or 0)
        ask   = float(quote[asym].ask_price or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return float(quote[asym].ask_price or quote[asym].bid_price or 100.0)
    except Exception:
        return 100.0


def compute_qty(symbol: str, kelly_fraction: float, equity: float, is_strong: bool) -> tuple:
    """Returns (qty, mid_price, limit_price). qty is always a whole int."""
    mid = get_mid_price(symbol)
    size_mult  = 1.0 if is_strong else 0.6
    kelly_boost = min(kelly_fraction * 2, MAX_RISK_PCT - BASE_RISK_PCT)
    risk_pct   = min(BASE_RISK_PCT + kelly_boost, MAX_RISK_PCT) * size_mult
    position_value = equity * risk_pct
    qty = max(int(position_value / mid), 1)   # int() here — never a float
    return qty, mid, mid


def get_existing_position_qty(symbol: str) -> float:
    asym = _alpaca_sym(symbol)
    try:
        return float(trading_client.get_open_position(asym).qty)
    except Exception:
        return 0.0


# ── P0: Stockformer Live Feature Builder ──────────────────────────────────

def _build_stockformer_features() -> dict:
    """
    Build real feature dict for Stockformer inference from daily bars.
    Returns {symbol: np.ndarray of shape [SEQ_LEN, N_FEATURES]}.
    """

    features_dict = {}
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=120)

        # Fetch SPY/QQQ for neutralization
        benchmarks = {}
        for bm in BENCHMARKS:
            req = StockBarsRequest(
                symbol_or_symbols=bm, timeframe=TimeFrame.Day,
                start=start, end=end, feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(bm, level=0)
            benchmarks[bm] = df.sort_index()["close"].values

        spy_c = benchmarks.get("SPY", np.array([]))
        qqq_c = benchmarks.get("QQQ", np.array([]))

        for symbol in WATCHLIST:
            try:
                sym_to_fetch = "TSM" if symbol == "TSMC" else symbol
                req = StockBarsRequest(
                    symbol_or_symbols=sym_to_fetch, timeframe=TimeFrame.Day,
                    start=start, end=end, feed=DataFeed.IEX,
                )
                bars = data_client.get_stock_bars(req)
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym_to_fetch, level=0)
                df = df.sort_index()

                closes = df["close"].values
                volumes = df["volume"].values
                n = len(closes)
                if n < SEQ_LEN:
                    continue

                # Build per-day features (each row = one trading day)
                feat_arr = np.zeros((n, N_FEATURES))

                rets = np.diff(closes, prepend=closes[0]) / (closes + 1e-8)

                # Feature 0: 5-day momentum (rolling, normalized)
                mom_5 = pd.Series(rets).rolling(5).sum().fillna(0).values
                mean_m, std_m = np.mean(mom_5), np.std(mom_5) + 1e-8
                feat_arr[:, 0] = (mom_5 - mean_m) / std_m

                # Feature 1: Volume (log-normalized)
                log_vol = np.log1p(volumes)
                mean_v, std_v = np.mean(log_vol), np.std(log_vol) + 1e-8
                feat_arr[:, 1] = (log_vol - mean_v) / std_v

                # Feature 2: Pure alpha vs SPY (rolling 5-day excess return)
                if len(spy_c) >= n:
                    spy_rets = np.diff(spy_c[:n], prepend=spy_c[0]) / (spy_c[:n] + 1e-8)
                    excess = rets - spy_rets
                    alpha_5 = pd.Series(excess).rolling(5).sum().fillna(0).values
                    mean_a, std_a = np.mean(alpha_5), np.std(alpha_5) + 1e-8
                    feat_arr[:, 2] = (alpha_5 - mean_a) / std_a

                # Feature 3: Realized vol (20-day rolling, normalized)
                rv = pd.Series(rets).rolling(20).std().fillna(0).values
                mean_rv, std_rv = np.mean(rv), np.std(rv) + 1e-8
                feat_arr[:, 3] = (rv - mean_rv) / std_rv

                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
                features_dict[symbol] = feat_arr

            except Exception as e:
                print(f"[STOCKFORMER] Feature build error for {symbol}: {e}")

    except Exception as e:
        print(f"[STOCKFORMER] Global feature build error: {e}")

    return features_dict


# ── P5: Order Fill Verification ──────────────────────────────────────────

def _verify_order_fill(alpaca_order_id: str, symbol: str):
    """Check IOC order status after 2 seconds and update Supabase."""
    try:
        time.sleep(2)
        order = trading_client.get_order_by_id(alpaca_order_id)
        status = str(order.status).lower()
        filled_qty = float(order.filled_qty or 0)
        filled_price = float(order.filled_avg_price or 0)

        if "filled" in status and filled_qty > 0:
            print(f"[FILL] {symbol}: FILLED {filled_qty} @ ${filled_price:.2f}")
            if USER_ID:
                supabase.table("trades").update({
                    "status": "filled",
                }).eq("alpaca_order_id", alpaca_order_id).execute()
        elif "cancel" in status or "expired" in status:
            print(f"[FILL] {symbol}: {status.upper()} — IOC order did not fill")
            if USER_ID:
                supabase.table("trades").update({
                    "status": "cancelled",
                }).eq("alpaca_order_id", alpaca_order_id).execute()
        elif "partial" in status:
            print(f"[FILL] {symbol}: PARTIAL fill {filled_qty} @ ${filled_price:.2f}")
            if USER_ID:
                supabase.table("trades").update({
                    "status": "partial",
                }).eq("alpaca_order_id", alpaca_order_id).execute()
        else:
            print(f"[FILL] {symbol}: status={status}")
    except Exception as e:
        print(f"[FILL] Verification error for {symbol}: {e}")


# ── Verdict Logic ─────────────────────────────────────────────────────────────

def referee_verdict(s: dict, m: dict) -> dict:
    s_dir, m_dir   = s["direction"], m["direction"]
    s_conf, m_conf = s["confidence"], m["confidence"]
    tug_score = round(abs(s_conf - m_conf), 4)

    if s_dir == "neutral" and m_dir == "neutral":
        return _result(s, m, False, "no_signal", tug_score)
    if s_dir == "neutral" or s_conf < SOVEREIGN_MIN_CONF:
        return _result(s, m, False, "no_signal", tug_score)

    # Both agree on same direction:
    # - HIGH confidence from both = strong consensus → EXECUTE
    # - Low confidence = crowded retail noise → skip
    if s_dir == m_dir:
        if s_conf >= SOVEREIGN_MIN_CONF and m_conf >= 0.65:
            return _result(s, m, False, "execute", tug_score)  # consensus signal
        return _result(s, m, False, "crowded_skip", tug_score)  # weak consensus

    # Classic tug: Sovereign vs Madman disagree → institutional edge
    if m_dir != "neutral" and s_dir != m_dir and s_conf >= SOVEREIGN_MIN_CONF:
        return _result(s, m, True, "execute", tug_score)
    if m_dir == "neutral" and s_conf >= SOVEREIGN_SOLO_CONF:
        return _result(s, m, True, "execute", tug_score)
    return _result(s, m, False, "no_signal", tug_score)


def _result(s, m, conflict, verdict, tug_score):
    return {
        "symbol": s["symbol"],
        "sovereign_direction": s["direction"],
        "madman_direction":    m["direction"],
        "conflict":  conflict,
        "verdict":   verdict,
        "sovereign_confidence": s["confidence"],
        "madman_confidence":    m["confidence"],
        "tug_score": tug_score,
    }


# ── Logging ───────────────────────────────────────────────────────────────────

def log_tug_result(result: dict) -> Optional[str]:
    if not USER_ID:
        return None
    try:
        resp = supabase.table("tug_results").insert({**result, "user_id": USER_ID}).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as e:
        print(f"[REFEREE] tug_results insert error: {e}")
        return None


# ── Crisis Hedging ───────────────────────────────────────────────────────────

def get_weakest_symbol() -> Optional[str]:
    """
    Find the watchlist symbol with the worst 5-day return.
    Used to pick the short target during Crisis.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta
    worst_sym  = None
    worst_ret  = float("inf")
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=10)
    for sym in WATCHLIST:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            df   = bars[sym].df if hasattr(bars[sym], "df") else bars.df
            if len(df) < 2:
                continue
            ret5 = (df["close"].iloc[-1] - df["close"].iloc[-min(5, len(df))]) / df["close"].iloc[-min(5, len(df))]
            if ret5 < worst_ret:
                worst_ret = ret5
                worst_sym = sym
        except Exception:
            continue
    print(f"[CRISIS] Weakest symbol: {worst_sym} (5d ret={worst_ret:.2%})")
    return worst_sym


def execute_crisis_hedges(equity: float, crisis_confidence: float):
    """
    Two-pronged Crisis hedge:
      1. Buy SQQQ (3× inverse QQQ) — profits as QQQ falls
      2. Short the weakest watchlist symbol — pure alpha short

    Only fires if:
      - Crisis confidence >= CRISIS_CONF_THRESHOLD (90%)
      - No existing SQQQ position already open
      - Market is open
    """
    if crisis_confidence < CRISIS_CONF_THRESHOLD:
        print(f"[CRISIS] Confidence {crisis_confidence:.2%} < {CRISIS_CONF_THRESHOLD:.0%} threshold — skipping hedges")
        return

    # ── 1. Inverse ETF hedge (SQQQ) ───────────────────────────
    sqqq_held = get_existing_position_qty(CRISIS_ETF)
    if sqqq_held > 0:
        print(f"[CRISIS] SQQQ hedge already open ({sqqq_held} shares) — skipping")
    else:
        mid = get_mid_price(CRISIS_ETF)
        qty = max(int((equity * CRISIS_ETF_RISK_PCT) / mid), 1)
        limit_price = round(mid * (1 + LIMIT_SLIP_PCT), 2)
        try:
            order = trading_client.submit_order(LimitOrderRequest(
                symbol=CRISIS_ETF,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.IOC,
                limit_price=limit_price,
            ))
            print(f"[CRISIS] SQQQ hedge: BUY {qty} @ ${limit_price} (IOC) | conf={crisis_confidence:.2%} | ID={order.id}")
            if USER_ID:
                supabase.table("trades").insert({
                    "symbol": CRISIS_ETF,
                    "side": "buy",
                    "qty": qty,
                    "order_type": "limit",
                    "order_type_detail": "crisis_hedge_sqqq",
                    "limit_price": limit_price,
                    "alpaca_order_id": str(order.id),
                    "status": "pending",
                    "implementation_shortfall_bps": round(abs(limit_price - mid) / mid * 10000, 2),
                    "user_id": USER_ID,
                }).execute()
        except Exception as e:
            print(f"[CRISIS] SQQQ order error: {e}")

    # ── 2. Short weakest watchlist symbol ─────────────────────
    short_sym = get_weakest_symbol()
    if not short_sym:
        print("[CRISIS] Could not determine weakest symbol — skipping short")
        return

    already_short = get_existing_position_qty(short_sym)
    if already_short < 0:
        print(f"[CRISIS] Already short {short_sym} — skipping")
        return
    if already_short > 0:
        print(f"[CRISIS] Long position in {short_sym} — closing before shorting")
        try:
            trading_client.close_position(short_sym)
        except Exception as e:
            print(f"[CRISIS] Could not close {short_sym} long: {e}")
            return

    mid   = get_mid_price(short_sym)
    qty   = max(int((equity * CRISIS_SHORT_RISK_PCT) / mid), 1)
    limit_price = round(mid * (1 - LIMIT_SLIP_PCT), 2)
    try:
        order = trading_client.submit_order(LimitOrderRequest(
            symbol=short_sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,   # shorts must be DAY orders on Alpaca paper
            limit_price=limit_price,
        ))
        print(f"[CRISIS] Short {short_sym}: SELL {qty} @ ${limit_price} (DAY) | conf={crisis_confidence:.2%} | ID={order.id}")
        if USER_ID:
            supabase.table("trades").insert({
                "symbol": short_sym,
                "side": "sell",
                "qty": qty,
                "order_type": "limit",
                "order_type_detail": "crisis_short_weakest",
                "limit_price": limit_price,
                "alpaca_order_id": str(order.id),
                "status": "pending",
                "implementation_shortfall_bps": round(abs(limit_price - mid) / mid * 10000, 2),
                "user_id": USER_ID,
            }).execute()
    except Exception as e:
        print(f"[CRISIS] Short order error for {short_sym}: {e}")


def close_crisis_hedges():
    """
    Called when regime flips OUT of Crisis.
    Closes SQQQ position and any active crisis shorts.
    """
    # Close SQQQ
    sqqq_held = get_existing_position_qty(CRISIS_ETF)
    if sqqq_held > 0:
        try:
            trading_client.close_position(CRISIS_ETF)
            print(f"[CRISIS] Closed SQQQ hedge ({sqqq_held} shares) — regime no longer Crisis")
        except Exception as e:
            print(f"[CRISIS] Error closing SQQQ: {e}")
    # Close any shorts on watchlist symbols (negative qty = short)
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if pos.symbol in WATCHLIST and float(pos.qty) < 0:
                try:
                    trading_client.close_position(pos.symbol)
                    print(f"[CRISIS] Closed crisis short: {pos.symbol} ({pos.qty} shares)")
                except Exception as e:
                    print(f"[CRISIS] Error closing short {pos.symbol}: {e}")
    except Exception as e:
        print(f"[CRISIS] Error fetching positions for short close: {e}")


# ── Server-side bracket orders ────────────────────────────────────────────────

def _place_bracket_orders(symbol: str, side: str, filled_price: float, qty: int, regime: str):
    """
    After a position fills, submit GTC stop-limit + GTC take-profit on Alpaca's servers.
    These execute even when the laptop is offline.

    Stop leg: StopLimitOrderRequest
      - stop_price  = entry ± stop_pct  (trigger level)
      - limit_price = stop_price ± 0.3% buffer (execution limit, avoids immediate trigger)
    Take leg: LimitOrderRequest at take_price (GTC)
    """
    stops = REGIME_STOPS.get(regime, REGIME_STOPS["trend"])
    stop_pct = stops["stop"]
    take_pct = stops["take"]

    if stop_pct == 0.0:
        return  # Crisis regime — no bracket

    # 0.3% execution buffer so the limit doesn't trigger on normal intraday noise
    STOP_LIMIT_BUFFER = 0.003

    asym = _alpaca_sym(symbol)
    try:
        if side == "buy":
            stop_trigger = round(filled_price * (1 - stop_pct), 2)
            stop_limit   = round(stop_trigger * (1 - STOP_LIMIT_BUFFER), 2)  # slightly below trigger
            take_price   = round(filled_price * (1 + take_pct), 2)
            exit_side    = OrderSide.SELL
        else:
            stop_trigger = round(filled_price * (1 + stop_pct), 2)
            stop_limit   = round(stop_trigger * (1 + STOP_LIMIT_BUFFER), 2)  # slightly above trigger
            take_price   = round(filled_price * (1 - take_pct), 2)
            exit_side    = OrderSide.BUY

        # Stop leg: proper StopLimitOrderRequest — only triggers when price crosses stop_trigger
        stop_req = StopLimitOrderRequest(
            symbol=asym,
            qty=qty,
            side=exit_side,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_trigger,
            limit_price=stop_limit,
        )
        # Take-profit leg: plain GTC limit at take_price
        take_req = LimitOrderRequest(
            symbol=asym,
            qty=qty,
            side=exit_side,
            time_in_force=TimeInForce.GTC,
            limit_price=take_price,
        )
        trading_client.submit_order(stop_req)
        trading_client.submit_order(take_req)
        print(f"[BRACKET] {asym}: stop_trigger=${stop_trigger} stop_limit=${stop_limit} ({stop_pct:.1%}) | take=${take_price} ({take_pct:.1%}) | regime={regime}")
    except Exception as e:
        print(f"[BRACKET] {asym}: bracket order error — {e}")


# ── Execution ─────────────────────────────────────────────────────────────────

def execute_trade(
    tug_result_id: Optional[str],
    symbol: str,
    side: str,
    kelly_fraction: float,
    equity: float,
    is_strong: bool = True,
    regime: str = "trend",
) -> Optional[dict]:

    qty, mid_price, _ = compute_qty(symbol, kelly_fraction, equity, is_strong)
    qty = max(int(qty), 1)  # CRITICAL: Limit IOC requires whole shares
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Sell guard: if selling, we must hold the position (paper can't naked short)
    if side == "sell":
        held = get_existing_position_qty(symbol)
        if held <= 0:
            print(f"[REFEREE] {symbol}: no position to sell, skipping")
            return None
        qty = max(int(min(held, qty)), 1)  # can't sell more than we hold

    # Limit price: mid ± slip% (aggressive enough to fill)
    if side == "buy":
        limit_price = round(mid_price * (1 + LIMIT_SLIP_PCT), 2)
    else:
        limit_price = round(mid_price * (1 - LIMIT_SLIP_PCT), 2)

    asym = _alpaca_sym(symbol)
    shortfall_bps = round(abs(limit_price - mid_price) / mid_price * 10000, 2)

    try:
        # Strategy: Try IOC first for best execution, fall back to DAY limit
        order_req = LimitOrderRequest(
            symbol=asym, qty=qty, side=order_side,
            time_in_force=TimeInForce.IOC, limit_price=limit_price,
        )
        order = trading_client.submit_order(order_req)
        alpaca_id = str(order.id)
        print(f"[REFEREE] IOC submitted: {side.upper()} {int(qty)} {asym} @ ${limit_price} | shortfall={shortfall_bps}bps | ID={alpaca_id}")

        # Check if IOC filled
        time.sleep(2)
        order_status = trading_client.get_order_by_id(alpaca_id)
        status_str = str(order_status.status).lower()
        filled_qty = float(order_status.filled_qty or 0)

        if "filled" in status_str and filled_qty > 0:
            filled_price = float(order_status.filled_avg_price or limit_price)
            real_shortfall = round(abs(filled_price - mid_price) / mid_price * 10000, 2)
            print(f"[FILL] {asym}: IOC FILLED {int(filled_qty)} @ ${filled_price:.2f} | shortfall={real_shortfall}bps")
            # Place server-side bracket (stop-market + take-profit) — survives offline
            _place_bracket_orders(symbol, side, filled_price, int(filled_qty), regime)
            if USER_ID:
                supabase.table("trades").insert({
                    "tug_result_id": tug_result_id, "symbol": symbol,
                    "side": side, "qty": int(filled_qty),
                    "order_type": "limit", "order_type_detail": "limit_ioc",
                    "limit_price": filled_price, "alpaca_order_id": alpaca_id,
                    "status": "filled", "implementation_shortfall_bps": real_shortfall,
                    "user_id": USER_ID,
                }).execute()
            return {"alpaca_order_id": alpaca_id, "qty": int(filled_qty), "side": side, "shortfall_bps": real_shortfall}

        # IOC cancelled/expired → fall back to DAY limit with wider slip
        print(f"[FILL] {asym}: IOC {status_str} — retrying as DAY limit")
        day_slip = 0.0025  # 0.25% — wider to guarantee fill
        if side == "buy":
            day_limit = round(mid_price * (1 + day_slip), 2)
        else:
            day_limit = round(mid_price * (1 - day_slip), 2)

        day_req = LimitOrderRequest(
            symbol=asym, qty=qty, side=order_side,
            time_in_force=TimeInForce.DAY, limit_price=day_limit,
        )
        day_order = trading_client.submit_order(day_req)
        day_id = str(day_order.id)
        day_shortfall = round(abs(day_limit - mid_price) / mid_price * 10000, 2)
        print(f"[REFEREE] DAY fallback: {side.upper()} {int(qty)} {asym} @ ${day_limit} | shortfall={day_shortfall}bps | ID={day_id}")

        if USER_ID:
            supabase.table("trades").insert({
                "tug_result_id": tug_result_id, "symbol": symbol,
                "side": side, "qty": int(qty),
                "order_type": "limit", "order_type_detail": "limit_day_fallback",
                "limit_price": day_limit, "alpaca_order_id": day_id,
                "status": "pending", "implementation_shortfall_bps": day_shortfall,
                "user_id": USER_ID,
            }).execute()
        return {"alpaca_order_id": day_id, "qty": int(qty), "side": side, "shortfall_bps": day_shortfall}

    except Exception as e:
        print(f"[REFEREE] Order error for {symbol}: {e}")
        if USER_ID:
            supabase.table("trades").insert({
                "tug_result_id": tug_result_id, "symbol": symbol,
                "side": side, "qty": int(qty),
                "order_type": "limit", "order_type_detail": "limit_ioc",
                "limit_price": limit_price, "status": "rejected",
                "user_id": USER_ID,
            }).execute()
        return None


# ── Main Cycle ────────────────────────────────────────────────────────────────

def run_cycle():
    print(f"\n{'='*60}")
    print(f"[REFEREE] Cycle start: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    if not is_market_open():
        print("[REFEREE] Market closed — skipping cycle")
        return

    # ── Connectivity check ────────────────────────────────────
    online = is_online()
    if not online:
        print("[OFFLINE] No internet — running in cache-only mode (server-side brackets protect positions)")
        cache = _load_cache()
        regime = cache["regime"]
        crisis_conf = cache["confidence"]
        equity = cache["equity"]
        print(f"[CACHE] regime={regime.upper()} equity=${equity:,.2f} — skipping new entries")
        # Still run exit checks with cached data so local stops fire if reconnected mid-cycle
        try:
            position_manager.run_exit_checks({}, {}, regime=regime)
        except Exception:
            pass
        return

    # ── Reconcile any orders that filled/cancelled while offline ─
    _reconcile_pending_orders()

    # ── Regime check ──────────────────────────────────────────
    regime_data = get_latest_regime_full()
    regime      = regime_data["state"]
    crisis_conf = regime_data["confidence"]
    print(f"[REFEREE] Regime: {regime.upper()} (conf={crisis_conf:.2%})")

    # Fix 1: Crisis streak — require CRISIS_STREAK_REQUIRED consecutive readings
    # before acting. Prevents single HMM noise spike from triggering $34 of churn.
    global _crisis_streak
    if regime == "crisis":
        _crisis_streak += 1
        print(f"[REFEREE] CRISIS streak: {_crisis_streak}/{CRISIS_STREAK_REQUIRED}")
        if _crisis_streak < CRISIS_STREAK_REQUIRED:
            print(f"[REFEREE] Crisis not confirmed yet — waiting for streak ({_crisis_streak}/{CRISIS_STREAK_REQUIRED})")
            # Still run exit checks on open positions, but don't deploy hedges yet
            equity = get_account_equity()
            position_manager.run_exit_checks({}, {}, regime="trend")
            return
        # Confirmed crisis — act
        print("[REFEREE] CRISIS confirmed — closing longs, deploying hedges")
        position_manager.run_exit_checks({}, {}, force_close=True)
        equity = get_account_equity()
        execute_crisis_hedges(equity, crisis_conf)
        return
    else:
        _crisis_streak = 0  # reset streak on any non-crisis reading

    # P2: Always clean up stale crisis hedges if regime is NOT crisis
    sqqq_held = get_existing_position_qty(CRISIS_ETF)
    if sqqq_held > 0:
        print(f"[REFEREE] Stale SQQQ hedge detected ({sqqq_held} shares) in {regime.upper()} — closing")
        close_crisis_hedges()

    equity = get_account_equity()
    print(f"[REFEREE] Account equity: ${equity:,.2f}")

    # ── Save last-known-good state for offline cycles ─────────
    _save_cache(regime, crisis_conf, equity)

    # ── Time-of-day filter ────────────────────────────────────
    if is_high_noise_window():
        print("[REFEREE] High-noise window (open/close 30min) — exits only, no new entries")
        position_manager.run_exit_checks({}, {}, regime=regime)
        return

    # ── Earnings filter ───────────────────────────────────────
    safe_symbols, blocked = filter_watchlist(WATCHLIST)
    if blocked:
        print(f"[REFEREE] Earnings block: {blocked}")

    # ── P0: Build Stockformer live features ──────────────────
    sf_features = _build_stockformer_features()
    sf_scores = stockformer_predict(sf_features)
    for sym, sc in sf_scores.items():
        if abs(sc) > 0.1:
            print(f"[STOCKFORMER] {sym}: conviction={sc:+.3f}")

    # ── Run bots ──────────────────────────────────────────────
    sovereign_results = sovereign_agent.run(safe_symbols, regime_state=regime)
    madman_results    = madman_agent.run(safe_symbols, regime_state=regime)

    s_map = {r["symbol"]: r for r in sovereign_results}
    m_map = {r["symbol"]: r for r in madman_results}

    # ── P1: Compute ensemble scores BEFORE execution ─────────
    regime_data_full = get_latest_regime_full()
    state_v3 = regime_data_full.get("state_v3", "")
    ensemble_map = {}
    for symbol in safe_symbols:
        sf = sf_scores.get(symbol, 0.0)
        m_result = m_map.get(symbol)
        ofi_z = m_result["raw_data"].get("ofi_z_score", 0.0) if m_result else 0.0
        iceberg = m_result["raw_data"].get("iceberg_detected", False) if m_result else False
        stacked = m_result["raw_data"].get("stacked_imbalance", False) if m_result else False
        trapped = m_result["raw_data"].get("trapped_exhaustion", False) if m_result else False
        ens = compute_ensemble_score(
            stockformer_score=sf, ofi_z=ofi_z, iceberg=iceberg,
            stacked=stacked, trapped=trapped,
            regime=regime, regime_confidence=crisis_conf,
            state_v3=state_v3,
        )
        ensemble_map[symbol] = ens
        if ens["ensemble_direction"] != "neutral":
            print(f"[META] {symbol}: {ens['ensemble_direction'].upper()} score={ens['ensemble_score']:.3f} (SF={ens['stockformer_component']:.3f} OFI={ens['ofi_component']:.3f} HMM={ens['hmm_component']:.3f})")

    # ── Exit checks first ─────────────────────────────────────
    position_manager.run_exit_checks(s_map, m_map, regime=regime)

    executed = skipped_crowded = skipped_no_signal = 0
    open_positions = get_open_position_count()
    print(f"[REFEREE] Open positions: {open_positions}/{MAX_OPEN_TRADES}")

    # Build list of currently held symbols for correlation check
    try:
        held_symbols = [p.symbol for p in trading_client.get_all_positions()]
    except Exception:
        held_symbols = []

    # Collect all execute candidates first, then filter by correlation
    execute_candidates = []
    for symbol in safe_symbols:
        s_result = s_map.get(symbol)
        m_result = m_map.get(symbol)
        if not s_result or not m_result:
            continue

        # FIX: Skip if we already hold this symbol (prevents 18x re-entry)
        alpaca_sym = _alpaca_sym(symbol)
        if alpaca_sym in held_symbols or symbol in held_symbols:
            print(f"[REFEREE] {symbol}: already held — skipping new entry")
            skipped_no_signal += 1
            continue

        verdict   = referee_verdict(s_result, m_result)
        tug_id    = log_tug_result(verdict)
        is_strong = m_result["direction"] != "neutral"
        edge_type = "STRONG" if is_strong else "WEAK"

        # P1: Ensemble override — if ensemble has a strong signal, use it
        ens = ensemble_map.get(symbol, {})
        ens_score = ens.get("ensemble_score", 0.0)
        ens_dir = ens.get("ensemble_direction", "neutral")

        print(
            f"[REFEREE] {symbol}: "
            f"Sovereign={verdict['sovereign_direction'].upper()}({verdict['sovereign_confidence']:.2%}) | "
            f"Madman={verdict['madman_direction'].upper()}({verdict['madman_confidence']:.2%}) | "
            f"TugScore={verdict['tug_score']:.2%} | "
            f"Ensemble={ens_score:+.3f} | "
            f"Verdict={verdict['verdict'].upper()}"
            + (f" [{edge_type}]" if verdict["verdict"] == "execute" else "")
        )

        should_execute = False
        exec_side = verdict["sovereign_direction"]

        # Original verdict logic
        if verdict["verdict"] == "execute":
            should_execute = True

        # P1: Ensemble override — strong ensemble signal can trigger execution
        # BUT must agree with Sovereign direction (don't fight the institutional signal)
        sov_dir = verdict["sovereign_direction"]
        if not should_execute and ens_score > ENSEMBLE_BUY_THRESHOLD and sov_dir in ("buy", "neutral"):
            should_execute = True
            exec_side = "buy"
            is_strong = True
            print(f"[REFEREE] {symbol}: ENSEMBLE OVERRIDE → BUY (score={ens_score:.3f})")
        elif not should_execute and ens_score < ENSEMBLE_SELL_THRESHOLD and sov_dir in ("sell", "neutral"):
            # Fix 5: Only SELL override if we actually hold the symbol
            if alpaca_sym in held_symbols or symbol in held_symbols:
                should_execute = True
                exec_side = "sell"
                is_strong = True
                print(f"[REFEREE] {symbol}: ENSEMBLE OVERRIDE → SELL (score={ens_score:.3f})")
            else:
                print(f"[REFEREE] {symbol}: ENSEMBLE SELL skipped — no position held")

        if should_execute:
            # Chop regime: only allow scalps (skip if not ORB-based signal)
            if regime == "chop":
                orb_dir = s_result["raw_data"].get("orb_breakout", "none")
                # P7: Also check VWAP deviation in chop
                if orb_dir == "none" and abs(ens_score) < 0.2:
                    print(f"[REFEREE] {symbol}: Chop regime — no ORB + weak ensemble, skipping")
                    skipped_no_signal += 1
                    continue
            execute_candidates.append((symbol, verdict, s_result, is_strong, exec_side))
        elif verdict["verdict"] == "crowded_skip":
            skipped_crowded += 1
        else:
            skipped_no_signal += 1

    # Apply correlation guard across all candidates
    candidate_pairs = [(sym, v) for sym, v, _, _, _ in execute_candidates]
    approved_pairs  = filter_by_correlation(candidate_pairs, held_symbols)
    approved_syms   = {sym for sym, _ in approved_pairs}

    for symbol, verdict, s_result, is_strong, exec_side in execute_candidates:
        if symbol not in approved_syms:
            print(f"[REFEREE] {symbol}: blocked by correlation guard")
            skipped_crowded += 1
            continue
        if open_positions >= MAX_OPEN_TRADES:
            print(f"[REFEREE] {symbol}: max positions reached, skipping")
            skipped_crowded += 1
            continue

        kelly_fraction = s_result["raw_data"].get("kelly_fraction", 0.02)
        result = execute_trade(
            tug_result_id=log_tug_result(verdict),
            symbol=symbol,
            side=exec_side,
            kelly_fraction=kelly_fraction,
            equity=equity,
            is_strong=is_strong,
            regime=regime,
        )
        if result:
            open_positions += 1
            executed += 1
            held_symbols.append(symbol)

    print(f"\n[REFEREE] Cycle complete — Executed: {executed} | Crowded Skip: {skipped_crowded} | No Signal: {skipped_no_signal} | Regime: {regime.upper()}")


# ── Background Tasks ──────────────────────────────────────────────────────────

def _run_scout_background():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scout_news_path = os.path.join(base, "scout", "scout_news.py")
    scout_alt_path  = os.path.join(base, "scout", "scout_alt.py")
    subprocess.Popen([sys.executable, scout_news_path], cwd=os.path.join(base, "scout"))
    subprocess.Popen([sys.executable, scout_alt_path],  cwd=os.path.join(base, "scout"))
    print("[REFEREE] Scout news + alt data refresh triggered")


def _run_hmm_background():
    """Re-fit HMM and log new regime state."""
    try:
        regime_hmm.run_and_log()
    except Exception as e:
        print(f"[REFEREE] HMM refresh error: {e}")


# ── Schedulers ────────────────────────────────────────────────────────────────

def run_once():
    run_cycle()


def run_scheduled(interval_minutes: int = 15):
    print(f"[REFEREE V2] Starting — cycles every {interval_minutes} min | Regime HMM every 60 min | Scout every 30 min")
    print(f"[REFEREE V2] Watchlist: {WATCHLIST}")

    # Initial HMM fit before first cycle
    _run_hmm_background()

    schedule.every(interval_minutes).minutes.do(run_cycle)
    schedule.every(30).minutes.do(_run_scout_background)
    schedule.every(60).minutes.do(_run_hmm_background)

    run_cycle()
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tug-Of-War Referee Engine V2")
    parser.add_argument("--once",     action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between cycles")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_scheduled(args.interval)
