"""
Position Manager — handles all exit logic for open trades.

Exit rules (in priority order):
1. STOP LOSS     : position down > 2% → exit immediately (hard risk limit)
2. TAKE PROFIT   : position up  > 4% → exit (lock in 2:1 reward/risk)
3. SIGNAL FLIP   : Sovereign flips direction on an open position → exit
4. CROWDED EXIT  : Madman joins our direction (trade became crowded) → exit
5. EOD CLOSE     : 15 min before market close → close all day trades

This is what separates a real quant bot from a signal generator.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pytz
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from supabase import create_client, Client

load_dotenv()

ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL     = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID          = os.getenv("USER_ID")
PAPER_TRADE      = os.getenv("ALPACA_PAPER_TRADE", "True").lower() == "true"

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADE)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

ET = pytz.timezone("America/New_York")

# Default stops (trend regime)
STOP_LOSS_PCT   = 0.02
TAKE_PROFIT_PCT = 0.04
EOD_CLOSE_MINS  = 15

# Regime-adjusted stops
REGIME_STOPS = {
    "trend":  {"stop": 0.02, "take": 0.04},
    "chop":   {"stop": 0.01, "take": 0.02},
    "crisis": {"stop": 0.00, "take": 0.00},
}


def get_open_positions() -> list:
    try:
        return trading_client.get_all_positions()
    except Exception as e:
        print(f"[POS_MGR] Error fetching positions: {e}")
        return []


def is_near_close() -> bool:
    now_et = datetime.now(ET)
    close_time = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return close_time <= now_et <= market_close


def close_position(symbol: str, reason: str, qty: Optional[float] = None):
    try:
        if qty is not None:
            req = ClosePositionRequest(qty=str(int(qty)))
            trading_client.close_position(symbol, close_options=req)
        else:
            trading_client.close_position(symbol)
        print(f"[POS_MGR] CLOSED {symbol} — reason: {reason}")
        _update_trade_record(symbol, reason)
    except Exception as e:
        print(f"[POS_MGR] Error closing {symbol}: {e}")


def _update_trade_record(symbol: str, exit_reason: str):
    if not USER_ID:
        return
    try:
        supabase.table("trades").update({
            "status": "filled",
        }).eq("symbol", symbol).eq("user_id", USER_ID).eq("status", "pending").execute()
    except Exception as e:
        print(f"[POS_MGR] Supabase update error: {e}")


def check_stop_take(position, regime: str = "trend") -> Optional[str]:
    try:
        stops = REGIME_STOPS.get(regime, REGIME_STOPS["trend"])
        unrealized_pct = float(position.unrealized_plpc)
        if unrealized_pct <= -stops["stop"]:
            return f"STOP_LOSS ({unrealized_pct:.2%}) [{regime}]"
        if unrealized_pct >= stops["take"]:
            return f"TAKE_PROFIT ({unrealized_pct:.2%}) [{regime}]"
    except Exception:
        pass
    return None


def check_signal_exit(position, sovereign_results: dict, madman_results: dict) -> Optional[str]:
    symbol = position.symbol
    side   = "buy" if float(position.qty) > 0 else "sell"

    s = sovereign_results.get(symbol)
    m = madman_results.get(symbol)
    if not s or not m:
        return None

    s_dir = s["direction"]
    m_dir = m["direction"]

    # Sovereign flipped against our position
    if side == "buy" and s_dir == "sell":
        return f"SIGNAL_FLIP (Sovereign now SELL)"
    if side == "sell" and s_dir == "buy":
        return f"SIGNAL_FLIP (Sovereign now BUY)"

    # Trade became crowded — Madman joined our direction (edge gone)
    if side == "buy" and m_dir == "buy":
        return f"CROWDED_EXIT (Madman now BUY — retail piled in)"
    if side == "sell" and m_dir == "sell":
        return f"CROWDED_EXIT (Madman now SELL — retail piled in)"

    return None


def run_exit_checks(
    sovereign_results: dict,
    madman_results: dict,
    regime: str = "trend",
    force_close: bool = False,
):
    positions = get_open_positions()

    if not positions:
        print("[POS_MGR] No open positions to check")
        return

    print(f"[POS_MGR] Checking {len(positions)} open position(s) | regime={regime}...")

    # Force close (crisis regime) — skip SQQQ hedge and any shorts (they're intentional)
    if force_close:
        print("[POS_MGR] Force close — crisis regime, closing long positions")
        for pos in positions:
            qty = float(pos.qty)
            if pos.symbol == "SQQQ":
                print(f"[POS_MGR] Keeping SQQQ crisis hedge — skipping")
                continue
            if qty < 0:
                print(f"[POS_MGR] Keeping short {pos.symbol} ({qty}) — crisis short, skipping")
                continue
            close_position(pos.symbol, "CRISIS_HALT")
        return

    # EOD close — 15 min before close, exit everything
    if is_near_close():
        print("[POS_MGR] Near market close — closing all positions (EOD rule)")
        for pos in positions:
            close_position(pos.symbol, "EOD_CLOSE")
        return

    for pos in positions:
        symbol = pos.symbol
        unrealized_pct = float(pos.unrealized_plpc)
        unrealized_pnl = float(pos.unrealized_pl)

        print(f"[POS_MGR] {symbol}: P&L={unrealized_pnl:+.2f} ({unrealized_pct:+.2%})")

        # 1. Stop loss / take profit (regime-adjusted)
        exit_reason = check_stop_take(pos, regime)
        if exit_reason:
            close_position(symbol, exit_reason)
            continue

        # 2. Signal-based exit
        exit_reason = check_signal_exit(pos, sovereign_results, madman_results)
        if exit_reason:
            close_position(symbol, exit_reason)
            continue

        print(f"[POS_MGR] {symbol}: holding — no exit trigger")
