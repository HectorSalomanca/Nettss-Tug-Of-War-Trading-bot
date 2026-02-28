"""
Scout 13F V3 — Institutional Telemetry from SEC EDGAR 13F Filings

Scrapes quarterly 13F-HR filings for top quantitative hedge funds:
  - Renaissance Technologies
  - Citadel Advisors
  - Two Sigma Investments
  - Millennium Management

Parses position changes → writes to Supabase `institutional_flows` table.
Sovereign agent applies confidence multiplier based on institutional alignment.

Runs weekly (13F filings are quarterly, but we check weekly for new filings).
"""

import os
import re
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup
import sys
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee.secret_vault import get_secret

SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

HEADERS = {
    "User-Agent": "TugOfWar Research Bot research@tugofwar.dev",
    "Accept": "application/json",
}

WATCHLIST = ["NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR"]

# CIK numbers for target funds (SEC EDGAR identifiers)
FUNDS = {
    "Renaissance Technologies": {"cik": "0001037389", "short": "RenTech"},
    "Citadel Advisors":         {"cik": "0001423053", "short": "Citadel"},
    "Two Sigma Investments":    {"cik": "0001179392", "short": "TwoSigma"},
    "Millennium Management":    {"cik": "0001273087", "short": "Millennium"},
}

# Ticker → CUSIP mapping for our watchlist (used to match 13F holdings)
TICKER_CUSIPS = {
    "NVDA":  "67066G104",
    "CRWD":  "22788C105",
    "LLY":   "532457108",
    "TSMC":  "874039100",  # TSM ADR
    "JPM":   "46625H100",
    "NEE":   "65339F101",
    "CAT":   "149123101",
    "SONY":  "835699307",  # SONY ADR
    "PLTR":  "69608A108",
    "MSTR":  "594972408",
}

CUSIP_TO_TICKER = {v: k for k, v in TICKER_CUSIPS.items()}


async def fetch_latest_13f(client: httpx.AsyncClient, fund_name: str, cik: str) -> Optional[dict]:
    """
    Fetch the most recent 13F-HR filing from SEC EDGAR for a given CIK.
    Returns parsed holdings for our watchlist tickers.
    """
    # EDGAR filing search API
    url = f"https://efts.sec.gov/LATEST/search-index?q=%2213F-HR%22&dateRange=custom&startdt=2025-01-01&enddt=2026-12-31&forms=13F-HR&ciks={cik}"
    fallback_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=5&search_text=&action=getcompany"

    try:
        # Try EDGAR full-text search first
        r = await client.get(
            f"https://efts.sec.gov/LATEST/search-index?q=&forms=13F-HR&ciks={cik}&dateRange=custom&startdt=2025-01-01",
            headers=HEADERS, timeout=15,
        )

        if r.status_code != 200:
            # Fallback: scrape EDGAR company page
            r = await client.get(fallback_url, headers=HEADERS, timeout=15)

        if r.status_code != 200:
            print(f"[13F] Failed to fetch filings for {fund_name}: HTTP {r.status_code}")
            return None

        # Parse the filing index to find the XML holdings file
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)[:10000]

        # Extract any holdings data we can find
        holdings = {}
        for ticker, cusip in TICKER_CUSIPS.items():
            # Search for CUSIP or company name in the filing text
            if cusip in text or ticker in text.upper():
                # Try to extract share count (rough heuristic)
                pattern = rf"(?:{cusip}|{ticker}).*?(\d{{1,3}}(?:,\d{{3}})*)"
                match = re.search(pattern, text, re.IGNORECASE)
                shares = int(match.group(1).replace(",", "")) if match else 0
                holdings[ticker] = shares

        if holdings:
            print(f"[13F] {fund_name}: found {len(holdings)} watchlist positions")
        return {"fund": fund_name, "holdings": holdings, "scraped_at": datetime.now(timezone.utc).isoformat()}

    except Exception as e:
        print(f"[13F] Error fetching {fund_name}: {e}")
        return None


def compute_flow_signal(current: dict, previous: dict) -> dict:
    """
    Compare current vs previous 13F holdings to detect accumulation/distribution.
    Returns per-ticker flow signals.
    """
    signals = {}
    current_holdings = current.get("holdings", {})
    prev_holdings = previous.get("holdings", {})

    for ticker in WATCHLIST:
        curr_shares = current_holdings.get(ticker, 0)
        prev_shares = prev_holdings.get(ticker, 0)

        if prev_shares == 0 and curr_shares == 0:
            continue

        if prev_shares == 0 and curr_shares > 0:
            action = "new_position"
            pct_change = 1.0
        elif curr_shares == 0 and prev_shares > 0:
            action = "exit"
            pct_change = -1.0
        else:
            pct_change = (curr_shares - prev_shares) / prev_shares if prev_shares > 0 else 0.0
            if pct_change > 0.1:
                action = "accumulation"
            elif pct_change < -0.1:
                action = "distribution"
            else:
                action = "hold"

        signals[ticker] = {
            "action": action,
            "pct_change": round(pct_change, 4),
            "current_shares": curr_shares,
            "previous_shares": prev_shares,
        }

    return signals


def log_institutional_flow(fund: str, ticker: str, flow: dict):
    """Write a single institutional flow record to Supabase."""
    if not USER_ID:
        return
    try:
        supabase.table("signals").insert({
            "symbol": ticker,
            "bot": "scout_13f",
            "direction": "buy" if flow["action"] in ("accumulation", "new_position") else
                         "sell" if flow["action"] in ("distribution", "exit") else "neutral",
            "confidence": min(0.5 + abs(flow["pct_change"]) * 0.3, 0.9),
            "signal_type": f"13f_{fund.lower().replace(' ', '_')}",
            "raw_data": {**flow, "fund": fund},
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        print(f"[13F] Supabase log error: {e}")


async def run_13f_scan():
    """Scan all target funds for 13F filings and log flow signals."""
    print(f"[13F] Starting 13F scan at {datetime.now(timezone.utc).isoformat()}")

    async with httpx.AsyncClient() as client:
        for fund_name, info in FUNDS.items():
            result = await fetch_latest_13f(client, fund_name, info["cik"])
            if result and result["holdings"]:
                # For now, log raw holdings (flow comparison requires previous quarter data)
                for ticker, shares in result["holdings"].items():
                    if shares > 0:
                        flow = {
                            "action": "holding",
                            "pct_change": 0.0,
                            "current_shares": shares,
                            "previous_shares": 0,
                        }
                        log_institutional_flow(fund_name, ticker, flow)
                        print(f"[13F] {info['short']}: {ticker} = {shares:,} shares")
            else:
                print(f"[13F] {info['short']}: no watchlist positions found")

    print("[13F] 13F scan complete")


def get_institutional_signal(symbol: str) -> dict:
    """
    Read most recent 13F signal from Supabase for a symbol.
    Used by sovereign_agent for confidence multiplier.
    """
    try:
        result = (
            supabase.table("signals")
            .select("direction, confidence, signal_type, raw_data, created_at")
            .eq("symbol", symbol)
            .eq("bot", "scout_13f")
            .order("created_at", desc=True)
            .limit(4)  # one per fund
            .execute()
        )
        if result.data:
            # Aggregate across funds
            buy_count = sum(1 for r in result.data if r["direction"] == "buy")
            sell_count = sum(1 for r in result.data if r["direction"] == "sell")
            avg_conf = float(sum(r["confidence"] for r in result.data) / len(result.data))

            if buy_count > sell_count:
                return {"direction": "buy", "confidence": avg_conf, "fund_count": len(result.data)}
            elif sell_count > buy_count:
                return {"direction": "sell", "confidence": avg_conf, "fund_count": len(result.data)}
    except Exception:
        pass
    return {"direction": "neutral", "confidence": 0.5, "fund_count": 0}


if __name__ == "__main__":
    asyncio.run(run_13f_scan())
