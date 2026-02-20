SOVEREIGN_SOURCES = [
    {
        "name": "SEC EDGAR Latest Filings",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F&dateb=&owner=include&count=10&search_text=",
        "category": "sovereign",
        "signal_type": "institutional_filing",
    },
    {
        "name": "Federal Reserve Press Releases",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases.htm",
        "category": "sovereign",
        "signal_type": "macro_policy",
    },
    {
        "name": "FinViz Insider Trading",
        "url": "https://finviz.com/insidertrading.ashx",
        "category": "sovereign",
        "signal_type": "insider_trade",
    },
    {
        "name": "MarketWatch Institutional News",
        "url": "https://www.marketwatch.com/investing/stocks",
        "category": "sovereign",
        "signal_type": "institutional_news",
    },
    {
        "name": "Seeking Alpha Earnings",
        "url": "https://seekingalpha.com/earnings/earnings-calendar",
        "category": "sovereign",
        "signal_type": "earnings",
    },
]

MADMAN_SOURCES = [
    {
        "name": "Reddit WallStreetBets Hot",
        "url": "https://www.reddit.com/r/wallstreetbets/hot/.json?limit=25",
        "category": "madman",
        "signal_type": "retail_sentiment",
    },
    {
        "name": "Reddit Stocks Hot",
        "url": "https://www.reddit.com/r/stocks/hot/.json?limit=25",
        "category": "madman",
        "signal_type": "retail_sentiment",
    },
    {
        "name": "FinViz News (Most Active)",
        "url": "https://finviz.com/news.ashx",
        "category": "madman",
        "signal_type": "retail_news",
    },
    {
        "name": "Yahoo Finance Trending",
        "url": "https://finance.yahoo.com/trending-tickers/",
        "category": "madman",
        "signal_type": "retail_trend",
    },
    {
        "name": "StockTwits Trending",
        "url": "https://api.stocktwits.com/api/2/trending/symbols.json",
        "category": "madman",
        "signal_type": "retail_sentiment",
    },
]

ALL_SOURCES = SOVEREIGN_SOURCES + MADMAN_SOURCES
