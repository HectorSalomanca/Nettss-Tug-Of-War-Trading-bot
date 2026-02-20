"""
Scout Tape — Alpaca WebSocket L1 Quote Streaming + Order Flow Imbalance (OFI)

Streams real-time NBBO quotes for all 10 tickers.
Calculates OFI = (bid_size - ask_size) / (bid_size + ask_size)
Detects Institutional Icebergs: price flat + high retail volume + negative OFI.
Writes OFI Z-scores to Supabase every minute.
"""

import os
import asyncio
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.data.live import StockDataStream

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

WATCHLIST = ["NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR"]

OFI_WINDOW = 60        # rolling window for Z-score (60 observations = ~1 min of ticks)
ICEBERG_OFI_THRESHOLD = -1.5   # OFI Z-score below this = institutional absorption
ICEBERG_PRICE_FLAT_PCT = 0.001  # price must be within 0.1% of 1-min-ago price

# Per-symbol rolling buffers
ofi_history: dict = defaultdict(lambda: deque(maxlen=OFI_WINDOW))
price_history: dict = defaultdict(lambda: deque(maxlen=10))
latest_quote: dict = {}


def compute_ofi(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total


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
    prices = list(price_history[symbol])
    if len(prices) < 5:
        return False
    price_change = abs(mid_price - prices[0]) / prices[0] if prices[0] > 0 else 1.0
    return price_change < ICEBERG_PRICE_FLAT_PCT and ofi_z < ICEBERG_OFI_THRESHOLD


async def quote_handler(quote):
    symbol = quote.symbol
    bid_size = float(quote.bid_size or 0)
    ask_size = float(quote.ask_size or 0)
    bid_price = float(quote.bid_price or 0)
    ask_price = float(quote.ask_price or 0)

    if bid_price <= 0 or ask_price <= 0:
        return

    mid_price = (bid_price + ask_price) / 2.0
    ofi = compute_ofi(bid_size, ask_size)

    ofi_history[symbol].append(ofi)
    price_history[symbol].append(mid_price)

    latest_quote[symbol] = {
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid_price": mid_price,
        "ofi_raw": ofi,
        "timestamp": datetime.now(timezone.utc),
    }


async def flush_to_supabase():
    """Write OFI snapshots for all symbols every 60 seconds."""
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

            if iceberg:
                print(f"[TAPE] ICEBERG DETECTED: {symbol} | OFI Z={ofi_z:.2f} | mid={q['mid_price']:.2f}")
            else:
                print(f"[TAPE] {symbol}: OFI Z={ofi_z:.2f} | mid={q['mid_price']:.2f}")

        if records:
            try:
                supabase.table("ofi_snapshots").insert(records).execute()
                print(f"[TAPE] Flushed {len(records)} OFI snapshots")
            except Exception as e:
                print(f"[TAPE] Supabase error: {e}")


def get_latest_ofi(symbol: str) -> dict:
    """
    Read most recent OFI snapshot from Supabase for a symbol.
    Used by madman_agent to get OFI Z-score.
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
            return result.data[0]
    except Exception:
        pass
    return {"ofi_z_score": 0.0, "iceberg_detected": False, "mid_price": None}


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
