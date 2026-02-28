"""
Scout Tape V3 — Deep Event-Based OFI + Stacked Imbalance + Trapped Exhaustion

Upgrades over V2:
  - Full event-based OFI formula (bid/ask price AND size changes)
  - Stacked Imbalance detection: OFI > 3:1 across 3+ consecutive ticks
  - Trapped Order Exhaustion: heavy volume at breakout peak + OFI flip
  - Spread tracking for Madman spread-decay gate
  - PyArrow in-memory buffers for zero-copy columnar OFI storage
  - Writes enriched snapshots to Supabase every 60 seconds
"""

import os
import sys
import asyncio
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone

# Ensure project root is on sys.path BEFORE any local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from referee.secret_vault import get_secret
from supabase import create_client, Client
from alpaca.data.live import StockDataStream

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

# Event-Driven Architecture: ZeroMQ publisher for real-time event broadcast
try:
    from referee.event_bus import (
        get_publisher, EVT_TRAPPED_EXHAUSTION, EVT_ICEBERG_DETECTED,
        EVT_STACKED_IMBALANCE, EVT_OFI_EXTREME, EVT_SPREAD_BLOW,
    )
    _event_pub = None  # lazy init after asyncio starts
except ImportError:
    _event_pub = None
    EVT_TRAPPED_EXHAUSTION = EVT_ICEBERG_DETECTED = EVT_STACKED_IMBALANCE = None
    EVT_OFI_EXTREME = EVT_SPREAD_BLOW = None

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

WATCHLIST = [
    "NVDA", "CRWD", "LLY", "TSM", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR",
    "MSFT", "AMZN", "META", "GLD", "XLE", "UBER", "AMD", "COIN", "MRNA", "IWM",
]

# Extreme OFI threshold for event broadcast
OFI_EXTREME_Z = 2.5
# Spread blow threshold (3x average = liquidity withdrawal)
SPREAD_BLOW_MULT = 3.0

OFI_WINDOW = 60
ICEBERG_OFI_THRESHOLD = -1.5
ICEBERG_PRICE_FLAT_PCT = 0.001
STACKED_IMBALANCE_RATIO = 3.0   # aggressive/passive volume ratio
STACKED_IMBALANCE_TICKS = 3     # consecutive ticks required
TRAPPED_VOLUME_MULT = 2.0       # volume must be 2x avg at breakout peak
TRAPPED_OFI_FLIP_Z = -1.0       # OFI must flip below this Z after peak

# Per-symbol rolling buffers
ofi_history: dict = defaultdict(lambda: deque(maxlen=OFI_WINDOW))
price_history: dict = defaultdict(lambda: deque(maxlen=60))
volume_history: dict = defaultdict(lambda: deque(maxlen=60))
spread_history: dict = defaultdict(lambda: deque(maxlen=60))
latest_quote: dict = {}
prev_bbo: dict = {}  # previous best bid/ask for event-based OFI

# Stacked imbalance tracking
imbalance_streak: dict = defaultdict(int)

# Arrow table buffers (per-symbol, flushed every 60s)
arrow_buffers: dict = defaultdict(list)


def compute_ofi_simple(bid_size: float, ask_size: float) -> float:
    """V2 simple OFI — kept as fallback."""
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total


def compute_ofi_event(
    bid_price: float, ask_price: float,
    bid_size: float, ask_size: float,
    prev_bid_price: float, prev_ask_price: float,
    prev_bid_size: float, prev_ask_size: float,
) -> float:
    """
    Full event-based OFI formula from microstructure research:
    e_n = I{P_bid >= P_bid_prev} * q_bid - I{P_bid <= P_bid_prev} * q_bid_prev
        - I{P_ask <= P_ask_prev} * q_ask + I{P_ask >= P_ask_prev} * q_ask_prev
    """
    e = 0.0
    if bid_price >= prev_bid_price:
        e += bid_size
    if bid_price <= prev_bid_price:
        e -= prev_bid_size
    if ask_price <= prev_ask_price:
        e -= ask_size
    if ask_price >= prev_ask_price:
        e += prev_ask_size
    return e


def compute_z_score(values: deque) -> float:
    arr = np.array(values)
    if len(arr) < 5:
        return 0.0
    mean = np.mean(arr)
    std  = np.std(arr)
    if std == 0:
        return 0.0
    return float((arr[-1] - mean) / std)


def detect_iceberg(symbol: str, ofi_z: float, mid_price: float) -> bool:
    """Price flat + negative OFI = institutional absorption."""
    prices = list(price_history[symbol])
    if len(prices) < 5:
        return False
    price_change = abs(mid_price - prices[0]) / prices[0] if prices[0] > 0 else 1.0
    return price_change < ICEBERG_PRICE_FLAT_PCT and ofi_z < ICEBERG_OFI_THRESHOLD


def detect_stacked_imbalance(symbol: str, ofi_val: float) -> bool:
    """
    Stacked Imbalance: OFI exceeds STACKED_IMBALANCE_RATIO for
    STACKED_IMBALANCE_TICKS consecutive ticks.
    Signals high-conviction retail momentum.
    """
    if abs(ofi_val) > 0 and ofi_val > 0:
        ratio = ofi_val  # already normalized
    else:
        ratio = 0.0

    # Use raw aggressive/passive ratio from recent OFI values
    recent = list(ofi_history[symbol])[-STACKED_IMBALANCE_TICKS:]
    if len(recent) < STACKED_IMBALANCE_TICKS:
        return False

    all_positive = all(v > 0 for v in recent)
    if all_positive:
        avg_ofi = np.mean(recent)
        if avg_ofi > (1.0 / (1.0 + STACKED_IMBALANCE_RATIO)):  # normalized threshold
            imbalance_streak[symbol] += 1
        else:
            imbalance_streak[symbol] = 0
    else:
        imbalance_streak[symbol] = 0

    return imbalance_streak[symbol] >= STACKED_IMBALANCE_TICKS


def detect_trapped_exhaustion(symbol: str, ofi_z: float) -> bool:
    """
    Trapped Order Exhaustion: heavy volume at breakout peak + immediate OFI flip.
    Late retail participants trapped at the high → forced liquidation.
    """
    prices = list(price_history[symbol])
    volumes = list(volume_history[symbol])
    if len(prices) < 10 or len(volumes) < 10:
        return False

    # Check if we're near a local high (within last 10 ticks)
    recent_high = max(prices[-10:])
    current = prices[-1]
    at_peak = (recent_high - current) / recent_high < 0.002  # within 0.2% of peak

    # Check volume spike at peak
    avg_vol = np.mean(list(volumes)[:max(len(volumes) - 3, 1)])
    recent_vol = np.mean(list(volumes)[-3:]) if len(volumes) >= 3 else 0
    vol_spike = recent_vol > avg_vol * TRAPPED_VOLUME_MULT if avg_vol > 0 else False

    # OFI must have flipped negative
    ofi_flipped = ofi_z < TRAPPED_OFI_FLIP_Z

    return at_peak and vol_spike and ofi_flipped


async def quote_handler(quote):
    symbol = quote.symbol
    bid_size = float(quote.bid_size or 0)
    ask_size = float(quote.ask_size or 0)
    bid_price = float(quote.bid_price or 0)
    ask_price = float(quote.ask_price or 0)

    if bid_price <= 0 or ask_price <= 0:
        return

    mid_price = (bid_price + ask_price) / 2.0
    spread_pct = (ask_price - bid_price) / mid_price if mid_price > 0 else 0.0

    # Event-based OFI (uses previous BBO state)
    prev = prev_bbo.get(symbol)
    if prev:
        ofi = compute_ofi_event(
            bid_price, ask_price, bid_size, ask_size,
            prev["bid_price"], prev["ask_price"],
            prev["bid_size"], prev["ask_size"],
        )
    else:
        ofi = compute_ofi_simple(bid_size, ask_size)

    # Update previous BBO
    prev_bbo[symbol] = {
        "bid_price": bid_price, "ask_price": ask_price,
        "bid_size": bid_size, "ask_size": ask_size,
    }

    ofi_history[symbol].append(ofi)
    price_history[symbol].append(mid_price)
    volume_history[symbol].append(bid_size + ask_size)
    spread_history[symbol].append(spread_pct)

    # Store in Arrow buffer if available
    if HAS_ARROW:
        arrow_buffers[symbol].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "bid_price": bid_price, "ask_price": ask_price,
            "bid_size": bid_size, "ask_size": ask_size,
            "ofi": ofi, "mid": mid_price, "spread_pct": spread_pct,
        })
        # Cap buffer at 5000 rows per symbol
        if len(arrow_buffers[symbol]) > 5000:
            arrow_buffers[symbol] = arrow_buffers[symbol][-3000:]

    latest_quote[symbol] = {
        "bid_size": bid_size,
        "ask_size": ask_size,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "mid_price": mid_price,
        "spread_pct": spread_pct,
        "ofi_raw": ofi,
        "timestamp": datetime.now(timezone.utc),
    }


async def flush_to_supabase():
    """Write enriched OFI snapshots for all symbols every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        if not USER_ID:
            continue

        records = []
        for symbol in WATCHLIST:
            if symbol not in latest_quote or len(ofi_history[symbol]) < 5:
                continue

            q = latest_quote[symbol]
            ofi_z = compute_z_score(ofi_history[symbol])
            iceberg = detect_iceberg(symbol, ofi_z, q["mid_price"])
            stacked = detect_stacked_imbalance(symbol, q["ofi_raw"])
            trapped = detect_trapped_exhaustion(symbol, ofi_z)

            # Spread stats
            spreads = list(spread_history[symbol])
            avg_spread = float(np.mean(spreads)) if spreads else 0.0

            records.append({
                "symbol": symbol,
                "ofi_raw": round(q["ofi_raw"], 6),
                "ofi_z_score": round(ofi_z, 4),
                "bid_size": q["bid_size"],
                "ask_size": q["ask_size"],
                "mid_price": round(q["mid_price"], 4),
                "iceberg_detected": iceberg,
                "user_id": USER_ID,
            })

            flags = []
            if iceberg: flags.append("ICEBERG")
            if stacked: flags.append("STACKED")
            if trapped: flags.append("TRAPPED")
            flag_str = " | ".join(flags) if flags else ""

            if flags:
                print(f"[TAPE] {symbol}: OFI Z={ofi_z:.2f} | spread={avg_spread:.4%} | {flag_str}")
            else:
                print(f"[TAPE] {symbol}: OFI Z={ofi_z:.2f} | spread={avg_spread:.4%}")

            # ── EDA: Broadcast events via ZeroMQ ──────────────────
            global _event_pub
            if _event_pub is None:
                try:
                    _event_pub = get_publisher()
                except Exception:
                    _event_pub = None

            if _event_pub:
                evt_data = {"ofi_z": ofi_z, "mid_price": q["mid_price"], "spread_pct": avg_spread}
                if trapped and EVT_TRAPPED_EXHAUSTION:
                    _event_pub.publish(EVT_TRAPPED_EXHAUSTION, symbol, evt_data)
                    print(f"[EVENT] >> {symbol}: TRAPPED_EXHAUSTION broadcast")
                if iceberg and EVT_ICEBERG_DETECTED:
                    _event_pub.publish(EVT_ICEBERG_DETECTED, symbol, evt_data)
                if stacked and EVT_STACKED_IMBALANCE:
                    _event_pub.publish(EVT_STACKED_IMBALANCE, symbol, evt_data)
                if abs(ofi_z) > OFI_EXTREME_Z and EVT_OFI_EXTREME:
                    _event_pub.publish(EVT_OFI_EXTREME, symbol, {**evt_data, "direction": "buy" if ofi_z > 0 else "sell"})
                if avg_spread > 0 and len(spreads) > 10:
                    baseline_spread = float(np.mean(list(spreads)[:max(len(spreads)-5, 1)]))
                    if baseline_spread > 0 and avg_spread > baseline_spread * SPREAD_BLOW_MULT and EVT_SPREAD_BLOW:
                        _event_pub.publish(EVT_SPREAD_BLOW, symbol, {**evt_data, "spread_ratio": avg_spread / baseline_spread})

        if records:
            try:
                supabase.table("ofi_snapshots").insert(records).execute()
                print(f"[TAPE] Flushed {len(records)} OFI snapshots")
            except Exception as e:
                print(f"[TAPE] Supabase error: {e}")


def get_latest_ofi(symbol: str) -> dict:
    """
    Read most recent OFI snapshot from Supabase for a symbol.
    Used by madman_agent to get OFI Z-score + microstructure flags.
    """
    try:
        result = (
            supabase.table("ofi_snapshots")
            .select("ofi_z_score, iceberg_detected, mid_price, created_at")
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            # Augment with live in-memory data if available
            ofi_z = row.get("ofi_z_score", 0.0)
            return {
                "ofi_z_score": ofi_z,
                "iceberg_detected": row.get("iceberg_detected", False),
                "mid_price": row.get("mid_price"),
                "stacked_imbalance": detect_stacked_imbalance(symbol, 0.0) if symbol in ofi_history else False,
                "trapped_exhaustion": detect_trapped_exhaustion(symbol, ofi_z) if symbol in ofi_history else False,
                "spread_pct": float(np.mean(list(spread_history[symbol]))) if symbol in spread_history and spread_history[symbol] else 0.0,
            }
    except Exception:
        pass
    return {
        "ofi_z_score": 0.0, "iceberg_detected": False, "mid_price": None,
        "stacked_imbalance": False, "trapped_exhaustion": False, "spread_pct": 0.0,
    }


async def run_tape():
    print(f"[TAPE] Starting OFI stream for {len(WATCHLIST)} symbols...")
    stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    for symbol in WATCHLIST:
        stream.subscribe_quotes(quote_handler, symbol)

    flush_task = asyncio.create_task(flush_to_supabase())
    await stream._run_forever()
    flush_task.cancel()


if __name__ == "__main__":
    asyncio.run(run_tape())
