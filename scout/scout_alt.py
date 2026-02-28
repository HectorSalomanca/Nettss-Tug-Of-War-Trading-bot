"""
Scout Alt Data V3 — Asset-Specific Alternative Data Scraper

Sources:
  - NVD CVE Database (CRWD) — zero-day / critical vulnerability alerts
  - FDA RSS (LLY) — drug approvals, fast-track designations
  - TSMC/NVDA supply chain — earnings call keyword scraping
  - General: 13F institutional flow signals (quarterly, via scout_13f.py)

Runs every 30 min alongside scout_news.py.
Outputs weighted sentiment vectors → Sovereign agent confidence multiplier.
"""

import os
import re
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import sys
import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee.secret_vault import get_secret
from supabase import create_client, Client

SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ── Per-Ticker Alt Data Config ───────────────────────────────────────────────

ALT_SOURCES = {
    "CRWD": {
        "name": "NVD CVE Database",
        "url": "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=crowdstrike&resultsPerPage=5",
        "type": "json",
        "bullish_keywords": ["patch deployed", "remediated", "fixed", "update available"],
        "bearish_keywords": ["zero-day", "critical", "actively exploited", "unpatched", "breach"],
        "weight": 3.0,  # high alpha for cybersecurity
    },
    "LLY": {
        "name": "FDA Press Releases",
        "url": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds",
        "fallback_url": "https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/novel-drug-approvals-2025",
        "type": "html",
        "bullish_keywords": ["approval", "approved", "fast-track", "breakthrough", "priority review",
                             "efficacy", "Phase 3 success", "tirzepatide", "GLP-1", "Eli Lilly"],
        "bearish_keywords": ["rejection", "complete response", "clinical hold", "safety concern",
                             "adverse event", "recall", "black box", "failed"],
        "weight": 4.0,  # FDA events cause violent moves
    },
    "NVDA": {
        "name": "TSMC Supply Chain",
        "url": "https://finance.yahoo.com/quote/TSM/news/",
        "type": "html",
        "bullish_keywords": ["CoWoS", "capacity expansion", "HBM", "AI chip", "data center",
                             "wafer yield", "N3", "record revenue", "strong demand"],
        "bearish_keywords": ["shortage", "delay", "yield issue", "export restriction",
                             "inventory correction", "weak demand", "capex cut"],
        "weight": 2.5,
    },
    "PLTR": {
        "name": "Government Contracts",
        "url": "https://sam.gov/search/?index=opp&sort=-modifiedDate&page=1&pageSize=5&sfm%5Bkeyword%5D=palantir",
        "fallback_url": "https://finance.yahoo.com/quote/PLTR/news/",
        "type": "html",
        "bullish_keywords": ["contract", "award", "DoD", "Army", "AIP", "Gotham",
                             "government", "defense", "intelligence", "NATO"],
        "bearish_keywords": ["protest", "cancelled", "delayed", "reduced", "lost bid"],
        "weight": 2.0,
    },
    "MSTR": {
        "name": "Bitcoin Treasury",
        "url": "https://finance.yahoo.com/quote/MSTR/news/",
        "type": "html",
        "bullish_keywords": ["bitcoin purchase", "BTC acquisition", "treasury", "convertible",
                             "Saylor", "hodl", "bitcoin reserve", "new high"],
        "bearish_keywords": ["sell", "liquidation", "margin call", "dilution",
                             "SEC investigation", "impairment", "loss"],
        "weight": 2.0,
    },
}


# ── Scraping Functions ───────────────────────────────────────────────────────

async def scrape_nvd_cve(client: httpx.AsyncClient) -> Optional[dict]:
    """Scrape NVD CVE database for CrowdStrike-related vulnerabilities."""
    config = ALT_SOURCES["CRWD"]
    try:
        r = await client.get(config["url"], headers=HEADERS, timeout=15)
        data = r.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return None

        text_blob = ""
        for v in vulns[:5]:
            cve = v.get("cve", {})
            desc = cve.get("descriptions", [{}])
            if desc:
                text_blob += desc[0].get("value", "") + " "

        return _score_text("CRWD", text_blob, config)
    except Exception as e:
        print(f"[ALT] NVD CVE error: {e}")
        return None


async def scrape_html_source(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Generic HTML scraper for FDA, Yahoo Finance, SAM.gov, etc."""
    config = ALT_SOURCES.get(symbol)
    if not config:
        return None

    url = config["url"]
    try:
        r = await client.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        if r.status_code != 200:
            # Try fallback URL
            fallback = config.get("fallback_url")
            if fallback:
                r = await client.get(fallback, headers=HEADERS, timeout=15, follow_redirects=True)

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)[:5000]

        if len(text) < 50:
            return None

        return _score_text(symbol, text, config)
    except Exception as e:
        print(f"[ALT] {symbol} scrape error: {e}")
        return None


def _score_text(symbol: str, text: str, config: dict) -> dict:
    """Score text against bullish/bearish keywords with weighting."""
    text_lower = text.lower()
    bull_hits = sum(1 for kw in config["bullish_keywords"] if kw.lower() in text_lower)
    bear_hits = sum(1 for kw in config["bearish_keywords"] if kw.lower() in text_lower)
    total = bull_hits + bear_hits

    if total == 0:
        direction = "neutral"
        confidence = 0.5
    elif bull_hits > bear_hits:
        direction = "buy"
        confidence = round(min(0.5 + (bull_hits / total) * 0.45, 0.95), 4)
    else:
        direction = "sell"
        confidence = round(min(0.5 + (bear_hits / total) * 0.45, 0.95), 4)

    # Apply source weight
    weighted_confidence = round(min(confidence * (config["weight"] / 2.0), 0.99), 4)

    return {
        "symbol": symbol,
        "source": config["name"],
        "direction": direction,
        "confidence": weighted_confidence,
        "bull_hits": bull_hits,
        "bear_hits": bear_hits,
        "weight": config["weight"],
        "text_snippet": text[:200],
    }


# ── Logging ──────────────────────────────────────────────────────────────────

def log_alt_signal(result: dict):
    if not USER_ID or not result:
        return
    try:
        supabase.table("signals").insert({
            "symbol": result["symbol"],
            "bot": "scout_alt",
            "direction": result["direction"],
            "confidence": result["confidence"],
            "signal_type": f"alt_data_{result['source'].lower().replace(' ', '_')}",
            "raw_data": result,
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        print(f"[ALT] Supabase log error: {e}")


# ── Main Runner ──────────────────────────────────────────────────────────────

async def run_all():
    """Scrape all alt data sources and log results."""
    print(f"[ALT] Starting alt data scan at {datetime.now(timezone.utc).isoformat()}")

    async with httpx.AsyncClient() as client:
        # CRWD: NVD CVE (JSON API)
        crwd_result = await scrape_nvd_cve(client)
        if crwd_result:
            print(f"[ALT] CRWD: {crwd_result['direction'].upper()} @ {crwd_result['confidence']:.2%} "
                  f"(bull={crwd_result['bull_hits']}, bear={crwd_result['bear_hits']})")
            log_alt_signal(crwd_result)

        # HTML sources: LLY, NVDA, PLTR, MSTR
        for symbol in ["LLY", "NVDA", "PLTR", "MSTR"]:
            result = await scrape_html_source(client, symbol)
            if result:
                print(f"[ALT] {symbol}: {result['direction'].upper()} @ {result['confidence']:.2%} "
                      f"(bull={result['bull_hits']}, bear={result['bear_hits']})")
                log_alt_signal(result)
            else:
                print(f"[ALT] {symbol}: no data")

    print(f"[ALT] Alt data scan complete")


def get_alt_signal(symbol: str) -> dict:
    """
    Read most recent alt data signal from Supabase for a symbol.
    Used by sovereign_agent to get alt data confidence multiplier.
    """
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence, signal_type, created_at")
            .eq("symbol", symbol)
            .eq("bot", "scout_alt")
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
            if age < 3600:  # only use if < 1 hour old
                return {
                    "direction": row["direction"],
                    "confidence": row["confidence"],
                    "signal_type": row["signal_type"],
                }
    except Exception:
        pass
    return {"direction": "neutral", "confidence": 0.5, "signal_type": "none"}


if __name__ == "__main__":
    asyncio.run(run_all())
