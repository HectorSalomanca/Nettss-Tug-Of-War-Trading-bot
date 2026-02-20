"""
Scout News — SEC 8-K filings (edgartools) + Yahoo/StockTitan headlines
with Bayesian Adversarial Filter to discard coordinated bot-farm signals.

Bayesian Filter logic:
  - If >= BOT_FARM_THRESHOLD social posts share the same phrasing
    BUT the SEC 8-K for that ticker is silent → discard as coordinated noise.
  - Bayesian score = P(real signal | social_volume, 8K_activity)
"""

import asyncio
import hashlib
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

WATCHLIST = ["NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR"]
BOT_FARM_THRESHOLD = 10   # posts sharing same phrase = suspicious

BULLISH_WORDS = ["beat", "surge", "rally", "breakout", "upgrade", "buy", "strong",
                 "growth", "record", "profit", "bullish", "outperform", "raise",
                 "positive", "upside", "momentum", "acquisition", "contract", "win"]
BEARISH_WORDS = ["miss", "drop", "fall", "downgrade", "sell", "weak", "loss",
                 "decline", "cut", "bearish", "underperform", "lower", "negative",
                 "lawsuit", "investigation", "recall", "warning", "risk", "debt"]


# ── Sentiment Scoring ─────────────────────────────────────────────────────────

def score_sentiment(text: str) -> tuple:
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    total = bull + bear
    if total == 0:
        return "neutral", 0.5
    if bull > bear:
        return "buy", round(0.5 + (bull / total) * 0.45, 4)
    return "sell", round(0.5 + (bear / total) * 0.45, 4)


def extract_phrase_fingerprints(text: str, n: int = 4) -> list:
    """Extract n-gram fingerprints for bot-farm detection."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


# ── SEC 8-K Fetcher ───────────────────────────────────────────────────────────

async def fetch_8k_sentiment(client: httpx.AsyncClient, ticker: str) -> tuple:
    """
    Fetch recent 8-K filings from SEC EDGAR RSS for a ticker.
    Returns (has_filing: bool, sentiment_direction, sentiment_confidence).
    """
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%228-K%22+%22{ticker}%22&dateRange=custom&startdt=2026-01-01&forms=8-K"
        headers = {"User-Agent": "TugOfWarBot research@tugofwar.ai"}
        r = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return False, "neutral", 0.5
        latest = hits[0].get("_source", {})
        description = latest.get("file_date", "") + " " + latest.get("display_names", [""])[0]
        direction, confidence = score_sentiment(description)
        return True, direction, confidence
    except Exception:
        return False, "neutral", 0.5


# ── Social Headline Fetcher ───────────────────────────────────────────────────

async def fetch_yahoo_headlines(client: httpx.AsyncClient, ticker: str) -> list:
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        r = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h3", "h2", "a"], limit=30):
            text = tag.get_text(strip=True)
            if len(text) > 20 and ticker.upper() in text.upper() or len(text) > 40:
                headlines.append(text)
        return headlines[:15]
    except Exception:
        return []


async def fetch_stocktwits(client: httpx.AsyncClient, ticker: str) -> list:
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = await client.get(url, timeout=10, follow_redirects=True)
        data = r.json()
        messages = data.get("messages", [])
        return [m.get("body", "") for m in messages[:20]]
    except Exception:
        return []


# ── Bayesian Adversarial Filter ───────────────────────────────────────────────

def bayesian_filter(
    social_texts: list,
    has_8k: bool,
    direction: str,
) -> tuple:
    """
    Returns (is_real_signal: bool, bayesian_score: float).

    Logic:
      P(real | 8K active) = 0.90
      P(real | 8K silent, low social dupe) = 0.65
      P(real | 8K silent, high social dupe) = 0.15  ← bot farm
    """
    if not social_texts:
        score = 0.65 if has_8k else 0.40
        return score > 0.5, round(score, 4)

    all_phrases = []
    for text in social_texts:
        all_phrases.extend(extract_phrase_fingerprints(text))

    phrase_counts = Counter(all_phrases)
    max_dupe = max(phrase_counts.values()) if phrase_counts else 0
    is_coordinated = max_dupe >= BOT_FARM_THRESHOLD

    if has_8k:
        score = 0.90
    elif is_coordinated:
        score = 0.15
        print(f"[SCOUT_NEWS] Bot-farm detected (max phrase dupe={max_dupe}) — discarding signal")
    else:
        score = 0.65

    return score > 0.5, round(score, 4)


# ── Per-Ticker Analysis ───────────────────────────────────────────────────────

async def analyze_ticker(client: httpx.AsyncClient, ticker: str) -> Optional[dict]:
    yahoo_headlines, stocktwits_posts, (has_8k, sec_dir, sec_conf) = await asyncio.gather(
        fetch_yahoo_headlines(client, ticker),
        fetch_stocktwits(client, ticker),
        fetch_8k_sentiment(client, ticker),
    )

    all_social = yahoo_headlines + stocktwits_posts
    combined_text = " ".join(all_social)

    if not combined_text.strip() and not has_8k:
        return None

    social_dir, social_conf = score_sentiment(combined_text)

    # 8-K overrides social if present
    if has_8k and sec_dir != "neutral":
        direction, confidence = sec_dir, sec_conf
    else:
        direction, confidence = social_dir, social_conf

    is_real, bayesian_score = bayesian_filter(all_social, has_8k, direction)

    if not is_real:
        print(f"[SCOUT_NEWS] {ticker}: DISCARDED by Bayesian filter (score={bayesian_score})")
        return None

    print(f"[SCOUT_NEWS] {ticker}: {direction.upper()} ({confidence:.2%}) | 8K={has_8k} | Bayes={bayesian_score}")

    return {
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "has_8k": has_8k,
        "bayesian_score": bayesian_score,
        "social_count": len(all_social),
    }


# ── Supabase Logging ──────────────────────────────────────────────────────────

def log_signal(ticker: str, direction: str, confidence: float,
               bayesian_score: float, has_8k: bool, regime_state: str = "trend"):
    if not USER_ID:
        return
    record = {
        "symbol": ticker,
        "bot": "sovereign",
        "direction": direction,
        "confidence": confidence,
        "signal_type": "8k_news_bayesian",
        "regime_state": regime_state,
        "bayesian_score": bayesian_score,
        "raw_data": {"has_8k": has_8k},
        "user_id": USER_ID,
    }
    try:
        supabase.table("signals").insert(record).execute()
    except Exception as e:
        print(f"[SCOUT_NEWS] Supabase error: {e}")


# ── Main Runner ───────────────────────────────────────────────────────────────

async def run_scout_news(regime_state: str = "trend") -> dict:
    print(f"\n[SCOUT_NEWS] Starting news scan for {len(WATCHLIST)} tickers...")
    results = {}

    async with httpx.AsyncClient() as client:
        tasks = {ticker: analyze_ticker(client, ticker) for ticker in WATCHLIST}
        for ticker, coro in tasks.items():
            result = await coro
            if result:
                log_signal(
                    ticker=ticker,
                    direction=result["direction"],
                    confidence=result["confidence"],
                    bayesian_score=result["bayesian_score"],
                    has_8k=result["has_8k"],
                    regime_state=regime_state,
                )
                results[ticker] = result

    print(f"[SCOUT_NEWS] Done. {len(results)}/{len(WATCHLIST)} tickers with valid signals.")
    return results


if __name__ == "__main__":
    asyncio.run(run_scout_news())
