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
import time
import schedule
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional
import pytz

from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.enums import DataFeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bots import sovereign_agent, madman_agent
from referee import position_manager
from quant import regime_hmm
from quant.earnings_filter import filter_watchlist
from quant.correlation_guard import filter_by_correlation
from quant.regime_hmm import get_latest_regime_full

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
CRISIS_ETF_RISK_PCT  = 0.03    # 3% equity into SQQQ hedge
CRISIS_SHORT_RISK_PCT = 0.02   # 2% equity into weakest-symbol short
CRISIS_CONF_THRESHOLD = 0.90   # only hedge if HMM is ≥90% confident in Crisis

# ── Position sizing ───────────────────────────────────────────────────────────
BASE_RISK_PCT   = 0.02    # 2% equity per trade (floor)
MAX_RISK_PCT    = 0.05    # 5% equity hard cap
MAX_OPEN_TRADES = 4

# ── Execution thresholds ──────────────────────────────────────────────────────
SOVEREIGN_MIN_CONF  = 0.75
SOVEREIGN_SOLO_CONF = 0.80
LIMIT_SLIP_PCT      = 0.0005   # limit price = mid ± 0.05% (aggressive, fills like market)

# ── Regime-adjusted stops ─────────────────────────────────────────────────────
REGIME_STOPS = {
    "trend":  {"stop": 0.02, "take": 0.04},
    "chop":   {"stop": 0.01, "take": 0.02},
    "crisis": {"stop": 0.00, "take": 0.00},   # crisis = no new trades
}

ET = pytz.timezone("America/New_York")


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


def get_open_position_count() -> int:
    try:
        return len(trading_client.get_all_positions())
    except Exception:
        return 0


def get_mid_price(symbol: str) -> float:
    try:
        req   = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quote = data_client.get_stock_latest_quote(req)
        bid   = float(quote[symbol].bid_price or 0)
        ask   = float(quote[symbol].ask_price or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return float(quote[symbol].ask_price or quote[symbol].bid_price or 100.0)
    except Exception:
        return 100.0


def compute_qty(symbol: str, kelly_fraction: float, equity: float, is_strong: bool) -> tuple:
    """Returns (qty, mid_price, limit_price)."""
    mid = get_mid_price(symbol)
    size_mult  = 1.0 if is_strong else 0.6
    kelly_boost = min(kelly_fraction * 2, MAX_RISK_PCT - BASE_RISK_PCT)
    risk_pct   = min(BASE_RISK_PCT + kelly_boost, MAX_RISK_PCT) * size_mult
    position_value = equity * risk_pct
    qty = max(int(position_value / mid), 1)   # Limit IOC requires whole shares
    return qty, mid, mid


def get_existing_position_qty(symbol: str) -> float:
    try:
        return float(trading_client.get_open_position(symbol).qty)
    except Exception:
        return 0.0


# ── Verdict Logic ─────────────────────────────────────────────────────────────

def referee_verdict(s: dict, m: dict) -> dict:
    s_dir, m_dir   = s["direction"], m["direction"]
    s_conf, m_conf = s["confidence"], m["confidence"]
    tug_score = round(abs(s_conf - m_conf), 4)

    if s_dir == "neutral" and m_dir == "neutral":
        return _result(s, m, False, "no_signal", tug_score)
    if s_dir == "neutral" or s_conf < SOVEREIGN_MIN_CONF:
        return _result(s, m, False, "no_signal", tug_score)
    if s_dir == m_dir:
        return _result(s, m, False, "crowded_skip", tug_score)
    if m_dir != "neutral" and s_dir != m_dir and s_conf >= SOVEREIGN_MIN_CONF:
        return _result(s, m, True, "execute", tug_score)
    if m_dir == "neutral" and s_conf >= SOVEREIGN_SOLO_CONF:
        return _result(s, m, True, "execute", tug_score)
    return _result(s, m, False, "crowded_skip", tug_score)


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
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Sell guard: paper accounts can't fractionally short
    if side == "sell":
        held = get_existing_position_qty(symbol)
        if held <= 0:
            print(f"[REFEREE] {symbol}: no position to sell, skipping")
            return None
        qty = max(float(int(min(held, qty))), 1.0)

    # Limit price: mid ± 0.05% (aggressive — fills like market, avoids slippage)
    if side == "buy":
        limit_price = round(mid_price * (1 + LIMIT_SLIP_PCT), 2)
    else:
        limit_price = round(mid_price * (1 - LIMIT_SLIP_PCT), 2)

    try:
        order_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.IOC,   # Immediate Or Cancel
            limit_price=limit_price,
        )
        order = trading_client.submit_order(order_req)
        alpaca_id = str(order.id)

        # Implementation shortfall = (limit - mid) / mid * 10000 bps
        shortfall_bps = round(abs(limit_price - mid_price) / mid_price * 10000, 2)

        print(f"[REFEREE] Order submitted: {side.upper()} {qty} {symbol} @ ${limit_price} (IOC) | shortfall={shortfall_bps}bps | ID={alpaca_id}")

        if USER_ID:
            supabase.table("trades").insert({
                "tug_result_id": tug_result_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "limit",
                "order_type_detail": "limit_ioc",
                "limit_price": limit_price,
                "alpaca_order_id": alpaca_id,
                "status": "pending",
                "implementation_shortfall_bps": shortfall_bps,
                "user_id": USER_ID,
            }).execute()

        return {"alpaca_order_id": alpaca_id, "qty": qty, "side": side, "shortfall_bps": shortfall_bps}

    except Exception as e:
        print(f"[REFEREE] Order error for {symbol}: {e}")
        if USER_ID:
            supabase.table("trades").insert({
                "tug_result_id": tug_result_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "limit",
                "order_type_detail": "limit_ioc",
                "limit_price": limit_price,
                "status": "rejected",
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

    # ── Regime check ──────────────────────────────────────────
    regime_data = get_latest_regime_full()
    regime      = regime_data["state"]
    crisis_conf = regime_data["confidence"]
    print(f"[REFEREE] Regime: {regime.upper()} (conf={crisis_conf:.2%})")

    if regime == "crisis":
        print("[REFEREE] CRISIS regime — closing longs, deploying hedges")
        position_manager.run_exit_checks({}, {}, force_close=True)
        equity = get_account_equity()
        execute_crisis_hedges(equity, crisis_conf)
        return

    # If we just exited Crisis, clean up any open hedges
    _prev_regime = getattr(run_cycle, "_last_regime", "trend")
    if _prev_regime == "crisis" and regime != "crisis":
        print("[REFEREE] Regime flipped out of Crisis — closing hedges")
        close_crisis_hedges()
    run_cycle._last_regime = regime

    equity = get_account_equity()
    print(f"[REFEREE] Account equity: ${equity:,.2f}")

    # ── Earnings filter ───────────────────────────────────────
    safe_symbols, blocked = filter_watchlist(WATCHLIST)
    if blocked:
        print(f"[REFEREE] Earnings block: {blocked}")

    # ── Run bots ──────────────────────────────────────────────
    sovereign_results = sovereign_agent.run(safe_symbols, regime_state=regime)
    madman_results    = madman_agent.run(safe_symbols, regime_state=regime)

    s_map = {r["symbol"]: r for r in sovereign_results}
    m_map = {r["symbol"]: r for r in madman_results}

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

        verdict   = referee_verdict(s_result, m_result)
        tug_id    = log_tug_result(verdict)
        is_strong = m_result["direction"] != "neutral"
        edge_type = "STRONG" if is_strong else "WEAK"

        print(
            f"[REFEREE] {symbol}: "
            f"Sovereign={verdict['sovereign_direction'].upper()}({verdict['sovereign_confidence']:.2%}) | "
            f"Madman={verdict['madman_direction'].upper()}({verdict['madman_confidence']:.2%}) | "
            f"TugScore={verdict['tug_score']:.2%} | "
            f"Verdict={verdict['verdict'].upper()}"
            + (f" [{edge_type}]" if verdict["verdict"] == "execute" else "")
        )

        if verdict["verdict"] == "execute":
            # Chop regime: only allow scalps (skip if not ORB-based signal)
            if regime == "chop":
                orb_dir = s_result["raw_data"].get("orb_breakout", "none")
                if orb_dir == "none":
                    print(f"[REFEREE] {symbol}: Chop regime — no ORB signal, skipping")
                    skipped_no_signal += 1
                    continue
            execute_candidates.append((symbol, verdict, s_result, is_strong))
        elif verdict["verdict"] == "crowded_skip":
            skipped_crowded += 1
        else:
            skipped_no_signal += 1

    # Apply correlation guard across all candidates
    candidate_pairs = [(sym, v) for sym, v, _, _ in execute_candidates]
    approved_pairs  = filter_by_correlation(candidate_pairs, held_symbols)
    approved_syms   = {sym for sym, _ in approved_pairs}

    for symbol, verdict, s_result, is_strong in execute_candidates:
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
            side=verdict["sovereign_direction"],
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
    scout_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scout", "scout_news.py"
    )
    subprocess.Popen([sys.executable, scout_path], cwd=os.path.dirname(scout_path))
    print("[REFEREE] Scout news refresh triggered")


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
