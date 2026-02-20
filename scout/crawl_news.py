import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client

from sources import ALL_SOURCES, SOVEREIGN_SOURCES, MADMAN_SOURCES

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

BULLISH_KEYWORDS = [
    "buy", "bull", "surge", "rally", "breakout", "upgrade", "beat", "strong",
    "growth", "record", "outperform", "accumulate", "long", "moon", "calls",
    "upside", "positive", "gains", "higher", "soar"
]
BEARISH_KEYWORDS = [
    "sell", "bear", "drop", "crash", "downgrade", "miss", "weak", "decline",
    "underperform", "short", "puts", "downside", "negative", "loss", "lower",
    "plunge", "risk", "warning", "cut", "reduce"
]

COMMON_WORDS = {
    # English words
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "ITS", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "DID", "OIL",
    "WILL", "WITH", "THIS", "THAT", "FROM", "THEY", "BEEN", "HAVE", "SAID",
    "EACH", "WHICH", "THEIR", "TIME", "ABOUT", "WOULD", "MAKE", "LIKE",
    "INTO", "THAN", "THEN", "SOME", "COULD", "WHEN", "WHAT", "WERE",
    # Finance acronyms / institutions (not tickers)
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "SEC", "FED", "GDP", "CPI",
    "USD", "EUR", "GBP", "JPY", "USA", "NYSE", "NASDAQ", "DOW", "IWM",
    "VIX", "FOMC", "FDIC", "CFTC", "FINRA", "IMF", "ECB", "BOJ", "BOE",
    "OPEC", "NATO", "WHO", "CDC", "FDA", "IRS", "DOJ", "FBI", "CIA",
    # Web / tech noise
    "HTTP", "HTTPS", "HTML", "JSON", "API", "URL", "RSS", "XML", "CSS",
    "EDGAR", "XBRL", "PDF", "FAQ", "TOS", "DMCA",
    # Market structure
    "SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "SLV", "USO", "UNG",
    "VWAP", "MACD", "RSI", "ATR", "EMA", "SMA", "OTC", "ATS",
    # Common 2-letter non-tickers
    "AI", "IT", "US", "UK", "EU", "AM", "PM", "ET", "PT", "CT",
    # Days / months
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
    "OCT", "NOV", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
}


def score_sentiment(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull_count + bear_count

    if total == 0:
        return "neutral", 0.5

    bull_ratio = bull_count / total
    bear_ratio = bear_count / total

    if bull_ratio > 0.6:
        confidence = min(0.5 + (bull_ratio - 0.5) * 1.5, 0.99)
        return "buy", round(confidence, 4)
    elif bear_ratio > 0.6:
        confidence = min(0.5 + (bear_ratio - 0.5) * 1.5, 0.99)
        return "sell", round(confidence, 4)
    else:
        return "neutral", 0.5


def extract_tickers(text: str, max_tickers: int = 5) -> list[str]:
    candidates = TICKER_PATTERN.findall(text)
    seen = set()
    tickers = []
    for t in candidates:
        if t not in COMMON_WORDS and t not in seen and len(t) >= 2:
            seen.add(t)
            tickers.append(t)
            if len(tickers) >= max_tickers:
                break
    return tickers


async def crawl_source(client: httpx.AsyncClient, source: dict) -> Optional[dict]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        # Reddit JSON endpoints don't need HTML parsing
        if source["url"].endswith(".json") or "api.stocktwits" in source["url"]:
            r = await client.get(source["url"], headers=headers, timeout=15, follow_redirects=True)
            text = r.text
        else:
            r = await client.get(source["url"], headers=headers, timeout=15, follow_redirects=True)
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

        if len(text) < 100:
            print(f"[SCOUT] Too short: {source['name']}")
            return None

        direction, confidence = score_sentiment(text)
        tickers = extract_tickers(text)

        print(f"[SCOUT] {source['name']}: {direction.upper()} ({confidence:.2%}) | tickers: {tickers[:3]}")

        return {
            "source": source,
            "markdown": text[:2000],
            "direction": direction,
            "confidence": confidence,
            "tickers": tickers,
        }
    except Exception as e:
        print(f"[SCOUT] Error on {source['name']}: {e}")
        return None


def log_signal(symbol: str, bot: str, direction: str, confidence: float,
               signal_type: str, raw_data: dict, source_url: str):
    if not USER_ID:
        print(f"[SCOUT] USER_ID not set — skipping Supabase write for {symbol}")
        return

    record = {
        "symbol": symbol,
        "bot": bot,
        "direction": direction,
        "confidence": confidence,
        "signal_type": signal_type,
        "raw_data": raw_data,
        "source_url": source_url,
        "user_id": USER_ID,
    }
    try:
        supabase.table("signals").insert(record).execute()
        print(f"[SCOUT] Logged signal: {bot.upper()} {direction.upper()} {symbol} ({confidence:.2%})")
    except Exception as e:
        print(f"[SCOUT] Supabase error: {e}")


async def run_scout():
    print(f"\n[SCOUT] Starting crawl at {datetime.now(timezone.utc).isoformat()}")
    print(f"[SCOUT] Sources: {len(ALL_SOURCES)} total ({len(SOVEREIGN_SOURCES)} sovereign, {len(MADMAN_SOURCES)} madman)\n")

    async with httpx.AsyncClient() as client:
        tasks = [crawl_source(client, src) for src in ALL_SOURCES]
        results = await asyncio.gather(*tasks)

    signals_logged = 0
    for result in results:
        if result is None:
            continue

        source = result["source"]
        tickers = result["tickers"]

        if not tickers:
            tickers = ["SPY"]

        for ticker in tickers[:3]:
            log_signal(
                symbol=ticker,
                bot=source["category"],
                direction=result["direction"],
                confidence=result["confidence"],
                signal_type=source["signal_type"],
                raw_data={
                    "source_name": source["name"],
                    "preview": result["markdown"][:500],
                    "all_tickers": result["tickers"],
                },
                source_url=source["url"],
            )
            signals_logged += 1

    print(f"\n[SCOUT] Done. {signals_logged} signals logged.")
    return signals_logged


if __name__ == "__main__":
    asyncio.run(run_scout())
