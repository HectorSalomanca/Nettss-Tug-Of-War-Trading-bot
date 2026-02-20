import os
import sys
import time
import schedule
from datetime import datetime, timezone, timedelta
from typing import Optional
import pytz

from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bots import sovereign, madman
from referee import position_manager

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID = os.getenv("USER_ID")
PAPER_TRADE = os.getenv("ALPACA_PAPER_TRADE", "True").lower() == "true"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADE)

WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "SPY", "QQQ"]

# ── Position sizing ──────────────────────────────────────────
# Base risk per trade: 2% of equity (casino house-edge model)
# Kelly multiplies this up to 5% max — never more
BASE_RISK_PCT   = 0.02   # 2% of equity per trade minimum
MAX_RISK_PCT    = 0.05   # 5% of equity hard cap
MAX_OPEN_TRADES = 4      # never hold more than 4 positions at once

# ── Execution thresholds ─────────────────────────────────────
# STRONG_EXECUTE : Sovereign ≥ 75% AND Madman actively opposes
#                  → highest edge, retail fighting smart money
# WEAK_EXECUTE   : Sovereign ≥ 80% AND Madman is neutral
#                  → retail hasn't piled in yet = best entry
# CROWDED_SKIP   : both bots agree direction (consensus = no edge)
# NO_SIGNAL      : both neutral
SOVEREIGN_MIN_CONF  = 0.75   # minimum sovereign conviction to trade
SOVEREIGN_SOLO_CONF = 0.80   # sovereign alone (madman neutral) threshold
ET = pytz.timezone("America/New_York")


def get_account_equity() -> float:
    try:
        account = trading_client.get_account()
        return float(account.equity)
    except Exception as e:
        print(f"[REFEREE] Account fetch error: {e}")
        return 10000.0


def compute_qty(symbol: str, kelly_fraction: float, equity: float, is_strong: bool) -> float:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quote = data_client.get_stock_latest_quote(req)
        price = float(quote[symbol].ask_price or quote[symbol].bid_price or 1)
    except Exception:
        price = 100.0

    # Casino model: base 2% risk, Kelly scales up to 5% max
    # Strong execute (active conflict) gets full size
    # Weak execute (sovereign solo) gets 60% size
    size_multiplier = 1.0 if is_strong else 0.6
    kelly_boost = min(kelly_fraction * 2, MAX_RISK_PCT - BASE_RISK_PCT)
    risk_pct = min(BASE_RISK_PCT + kelly_boost, MAX_RISK_PCT) * size_multiplier
    position_value = equity * risk_pct
    qty = position_value / price
    return max(round(qty, 4), 0.01)


def is_market_open() -> bool:
    """True during regular hours (9:30–16:00 ET, Mon–Fri) or pre/post market."""
    try:
        account = trading_client.get_account()
        _ = account  # just check connectivity
    except Exception:
        pass
    now_et = datetime.now(ET)
    weekday = now_et.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:
        return False
    market_open  = now_et.replace(hour=4,  minute=0,  second=0, microsecond=0)
    market_close = now_et.replace(hour=20, minute=0,  second=0, microsecond=0)
    return market_open <= now_et <= market_close


def get_open_position_count() -> int:
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception:
        return 0


def referee_verdict(sovereign_result: dict, madman_result: dict) -> dict:
    s_dir  = sovereign_result["direction"]
    m_dir  = madman_result["direction"]
    s_conf = sovereign_result["confidence"]
    m_conf = madman_result["confidence"]

    tug_score = round(abs(s_conf - m_conf), 4)

    # ── Both neutral → no signal ──────────────────────────────
    if s_dir == "neutral" and m_dir == "neutral":
        return _result(sovereign_result, madman_result, False, "no_signal", tug_score)

    # ── Sovereign has no conviction → skip ───────────────────
    if s_dir == "neutral" or s_conf < SOVEREIGN_MIN_CONF:
        return _result(sovereign_result, madman_result, False, "no_signal", tug_score)

    # ── Both agree direction → crowded trade, no edge ────────
    if s_dir == m_dir:
        return _result(sovereign_result, madman_result, False, "crowded_skip", tug_score)

    # ── STRONG EXECUTE: Sovereign ≥ 75% + Madman actively opposes
    #    Retail is fighting smart money → maximum edge
    if m_dir != "neutral" and s_dir != m_dir and s_conf >= SOVEREIGN_MIN_CONF:
        return _result(sovereign_result, madman_result, True, "execute", tug_score)

    # ── WEAK EXECUTE: Sovereign ≥ 80% + Madman neutral
    #    Retail hasn't piled in yet → good entry, smaller size
    if m_dir == "neutral" and s_conf >= SOVEREIGN_SOLO_CONF:
        return _result(sovereign_result, madman_result, True, "execute", tug_score)

    # ── Fallback ──────────────────────────────────────────────
    return _result(sovereign_result, madman_result, False, "crowded_skip", tug_score)


def _result(s: dict, m: dict, conflict: bool, verdict: str, tug_score: float) -> dict:
    return {
        "symbol": s["symbol"],
        "sovereign_direction": s["direction"],
        "madman_direction": m["direction"],
        "conflict": conflict,
        "verdict": verdict,
        "sovereign_confidence": s["confidence"],
        "madman_confidence": m["confidence"],
        "tug_score": tug_score,
    }


def log_tug_result(result: dict) -> Optional[str]:
    if not USER_ID:
        print(f"[REFEREE] USER_ID not set — skipping Supabase write")
        return None
    record = {**result, "user_id": USER_ID}
    try:
        resp = supabase.table("tug_results").insert(record).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as e:
        print(f"[REFEREE] tug_results insert error: {e}")
        return None


def get_existing_position_qty(symbol: str) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def execute_trade(tug_result_id: Optional[str], symbol: str, side: str,
                  kelly_fraction: float, equity: float, is_strong: bool = True) -> Optional[dict]:
    qty = compute_qty(symbol, kelly_fraction, equity, is_strong)
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Alpaca paper does not allow fractional short selling
    # For sells: only sell if we hold the position, use whole shares
    if side == "sell":
        held_qty = get_existing_position_qty(symbol)
        if held_qty <= 0:
            print(f"[REFEREE] {symbol}: no position to sell, skipping short")
            return None
        qty = min(float(int(held_qty)), qty)  # whole shares only, cap at held
        qty = max(round(qty, 0), 1.0)

    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_request)
        alpaca_order_id = str(order.id)
        print(f"[REFEREE] Order submitted: {side.upper()} {qty} {symbol} | Alpaca ID: {alpaca_order_id}")

        if USER_ID:
            trade_record = {
                "tug_result_id": tug_result_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "market",
                "alpaca_order_id": alpaca_order_id,
                "status": "pending",
                "user_id": USER_ID,
            }
            supabase.table("trades").insert(trade_record).execute()

        return {"alpaca_order_id": alpaca_order_id, "qty": qty, "side": side}

    except Exception as e:
        print(f"[REFEREE] Order error for {symbol}: {e}")
        if USER_ID:
            trade_record = {
                "tug_result_id": tug_result_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "market",
                "status": "rejected",
                "user_id": USER_ID,
            }
            supabase.table("trades").insert(trade_record).execute()
        return None


def run_cycle():
    print(f"\n{'='*60}")
    print(f"[REFEREE] Cycle start: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    if not is_market_open():
        print("[REFEREE] Market closed — skipping cycle")
        return

    equity = get_account_equity()
    print(f"[REFEREE] Account equity: ${equity:,.2f}")

    sovereign_results = sovereign.run(WATCHLIST)
    madman_results = madman.run(WATCHLIST)

    # ── EXIT CHECKS: run before entering new positions ──────
    s_map_for_exit = {r["symbol"]: r for r in sovereign_results}
    m_map_for_exit = {r["symbol"]: r for r in madman_results}
    position_manager.run_exit_checks(s_map_for_exit, m_map_for_exit)

    sovereign_map = {r["symbol"]: r for r in sovereign_results}
    madman_map = {r["symbol"]: r for r in madman_results}

    executed = 0
    skipped_crowded = 0
    skipped_no_signal = 0

    open_positions = get_open_position_count()
    print(f"[REFEREE] Open positions: {open_positions}/{MAX_OPEN_TRADES}")

    for symbol in WATCHLIST:
        s_result = sovereign_map.get(symbol)
        m_result = madman_map.get(symbol)

        if not s_result or not m_result:
            continue

        verdict = referee_verdict(s_result, m_result)
        tug_result_id = log_tug_result(verdict)

        is_strong = m_result["direction"] != "neutral"
        edge_type = "STRONG" if is_strong else "WEAK"

        print(
            f"[REFEREE] {symbol}: "
            f"Sovereign={verdict['sovereign_direction'].upper()}({verdict['sovereign_confidence']:.2%}) | "
            f"Madman={verdict['madman_direction'].upper()}({verdict['madman_confidence']:.2%}) | "
            f"TugScore={verdict['tug_score']:.2%} | "
            f"Verdict={verdict['verdict'].upper()}"
            + (f" [{edge_type}]" if verdict['verdict'] == 'execute' else "")
        )

        if verdict["verdict"] == "execute":
            if open_positions >= MAX_OPEN_TRADES:
                print(f"[REFEREE] {symbol}: max positions reached, skipping")
                skipped_crowded += 1
                continue
            kelly_fraction = s_result["raw_data"].get("kelly_fraction", 0.02)
            execute_trade(
                tug_result_id=tug_result_id,
                symbol=symbol,
                side=verdict["sovereign_direction"],
                kelly_fraction=kelly_fraction,
                equity=equity,
                is_strong=is_strong,
            )
            open_positions += 1
            executed += 1
        elif verdict["verdict"] == "crowded_skip":
            skipped_crowded += 1
        else:
            skipped_no_signal += 1

    print(f"\n[REFEREE] Cycle complete — Executed: {executed} | Crowded Skip: {skipped_crowded} | No Signal: {skipped_no_signal}")


def run_once():
    run_cycle()


def run_scheduled(interval_minutes: int = 15):
    print(f"[REFEREE] Scheduling cycles every {interval_minutes} minutes (market hours only)...")
    print(f"[REFEREE] Scout will run every 30 minutes to refresh sentiment data")
    schedule.every(interval_minutes).minutes.do(run_cycle)
    # Scout refreshes sentiment every 30 min independently
    schedule.every(30).minutes.do(_run_scout_background)
    run_cycle()
    while True:
        schedule.run_pending()
        time.sleep(30)


def _run_scout_background():
    import subprocess
    scout_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scout", "crawl_news.py")
    subprocess.Popen([sys.executable, scout_path], cwd=os.path.dirname(scout_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tug-Of-War Referee Engine")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between cycles (default: 15)")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_scheduled(args.interval)
