# Tug-Of-War V3 — Institutional-Grade Quantitative Trading Architecture

## Complete Technical Reference

This document provides an exhaustive, in-depth explanation of every component in the Tug-Of-War V3 system — a dual-agent quantitative trading engine that exploits the behavioral friction between institutional capital and retail momentum. The system runs 24/7 on an M1 MacBook Pro via `launchd`, executes through Alpaca paper trading, stores all telemetry in Supabase, and uses a Next.js dashboard for real-time monitoring.

---

## Table of Contents

1. [System Overview & Philosophy](#1-system-overview--philosophy)
2. [The Sovereign Agent (Institutional Bot)](#2-the-sovereign-agent)
3. [The Madman Agent (Retail Bot)](#3-the-madman-agent)
4. [The Scout Layer (Data Ingestion)](#4-the-scout-layer)
5. [The Quant Layer (Signal Processing)](#5-the-quant-layer)
6. [The Referee Engine (Decision Layer)](#6-the-referee-engine)
7. [Position Management & Risk Controls](#7-position-management--risk-controls)
8. [Infrastructure & Deployment](#8-infrastructure--deployment)
9. [Database Schema](#9-database-schema)
10. [Mathematical Foundations](#10-mathematical-foundations)
11. [Supabase Schema Migration (V3)](#11-supabase-schema-migration-v3)

---

## 1. System Overview & Philosophy

### Core Thesis
The maximum mathematical edge in modern markets exists at the **point of maximum disagreement** between institutional capital and retail momentum. When the retail crowd aggressively chases price action into a zone of institutional liquidity absorption, a predictable reversal follows. The Tug-Of-War system quantifies this disagreement every 15 minutes and executes when the tension exceeds calibrated thresholds.

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        REFEREE ENGINE V2                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Sovereign    │    │   Madman      │    │  Meta-Model      │   │
│  │  Agent V3     │◄──►│   Agent V3    │───►│  Ensemble        │   │
│  │  (Hedge Fund) │    │   (Retail)    │    │  (Ridge-weighted)│   │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘   │
│         │                   │                      │             │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌────────▼─────────┐   │
│  │ Stockformer   │    │ Deep L2 OFI  │    │ 4-State HMM      │   │
│  │ Transformer   │    │ Event-Based  │    │ Regime Switching  │   │
│  │ (MPS GPU)     │    │ (Arrow)      │    │ (Bull/Bear/Chop)  │   │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘   │
│         │                   │                      │             │
│  ┌──────▼───────────────────▼──────────────────────▼─────────┐   │
│  │              QUANT LAYER (Feature Factory)                 │   │
│  │  AFD · Pure Alpha · Ticker Profiles · Correlation Guard    │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │              SCOUT LAYER (Data Ingestion)                  │   │
│  │  News · 8-K · Alt Data · 13F · OFI Tape · Bayesian Filter │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
    ┌────▼────┐                              ┌──────▼──────┐
    │ Alpaca  │                              │  Supabase   │
    │ Trading │                              │  (Postgres) │
    └─────────┘                              └─────────────┘
```

### Watchlist
| Ticker | Sector | Why Selected |
|--------|--------|-------------|
| NVDA | Tech Growth | AI chip leader, highest institutional flow, TSMC supply chain proxy |
| CRWD | Tech Growth | Cybersecurity leader, CVE-driven events, high retail interest |
| LLY | Healthcare | GLP-1/tirzepatide leader, FDA event-driven, institutional accumulation |
| TSMC | International | Semiconductor foundry monopoly, NVDA supply chain, macro proxy |
| JPM | Financials | Banking bellwether, NII-driven, low beta anchor |
| NEE | Utilities | Renewable energy leader, rate-case driven, defensive anchor |
| CAT | Industrials | Infrastructure proxy, tariff-sensitive, cyclical indicator |
| SONY | International | Gaming/entertainment diversifier, yen-sensitive |
| PLTR | Tech Growth | Government AI contracts, high beta, retail favorite |
| MSTR | Crypto Proxy | Bitcoin treasury proxy, extreme beta, leverage indicator |

**Silent Benchmarks:** SPY, QQQ (used for neutralization, never traded)

---

## 2. The Sovereign Agent

**File:** `bots/sovereign_agent.py`
**Role:** Mimics institutional/hedge-fund logic. High-conviction, fundamental-driven.

### Signal Stack (V3)

#### 2.1 Adaptive Fractional Differencing (AFD)
**File:** `quant/feature_factory.py`

Standard integer differencing (daily returns) destroys long-term memory in price series. AFD uses the fractional integration operator:

```
(1 - B)^d X_t = Σ_{k=0}^{∞} (-1)^k C(d,k) B^k X_t
```

Where `d` is a real number in `[0, 1]` found by binary search — the minimum `d` that makes the series stationary (ADF test p < 0.05) while preserving maximum trend memory.

**Implementation:**
- `_get_weights(d, size)` — computes fracDiff weights via binomial recursion
- `frac_diff(series, d)` — applies weights with cutoff threshold 1e-5
- `adaptive_frac_diff(prices)` — searches d ∈ [0, 1] in 0.1 steps, returns (differenced_series, d_value)
- `build_features()` — combines AFD momentum + SPY/QQQ-neutralized pure alpha

**Weight formula:** `w_k = -w_{k-1} * (d - k + 1) / k`

#### 2.2 Stockformer Transformer (V3)
**File:** `quant/stockformer.py`

A multi-asset Transformer neural network that processes all 10 tickers simultaneously, capturing cross-sectional lead-lag relationships.

**Architecture:**
```
Input: [batch, 60 days, 10 assets, 4 features]
  ↓
Input Projection: Linear(4 → 64) + Positional Embedding + Asset Embedding
  ↓
StockformerBlock × 2:
  ├── Temporal Self-Attention (per asset, across time)
  ├── Cross-Asset Attention (per timestep, across assets)
  │   └── Learnable Lead-Lag Bias Matrix (NVDA→CRWD: 0.3, TSMC→MSTR: 0.2, etc.)
  └── Feed-Forward Network (64 → 256 → 64, GELU activation)
  ↓
Pool last timestep → Linear(64 → 32 → 1) → Tanh
  ↓
Output: [batch, 10] conviction scores in [-1, 1]
```

**Key innovations:**
- **Negative Sharpe Loss:** Trains directly on risk-adjusted returns, not MSE
- **Lead-Lag Bias Matrix:** Hard-coded attention bias for known cross-sectional relationships
- **MPS Backend:** Runs on M1 GPU via `torch.device("mps")` — 3-5× faster than CPU
- **Early Stopping:** Patience=10 epochs, saves best model to `models/stockformer.pt`

**4 Input Features per asset per day:**
1. AFD-differenced close (normalized)
2. Log-normalized volume
3. Pure alpha (SPY/QQQ-neutralized return)
4. 20-day realized volatility

**Cross-Sectional Lead-Lag Pairs:**
| Leader | Lagger | Bias Weight | Mechanism |
|--------|--------|-------------|-----------|
| NVDA (0) | CRWD (1) | 0.30 | Tech stack capital flow: hardware → software |
| TSMC (3) | MSTR (9) | 0.20 | Macro liquidity constraint correlation |
| NVDA (0) | TSMC (3) | 0.15 | Supply chain: customer → foundry |
| PLTR (8) | MSTR (9) | 0.10 | High-beta correlation cluster |

**Training:** Weekly retraining on 1-year rolling window. 213 samples from 274 trading days. Model size: ~579KB.

#### 2.3 Opening Range Breakout (ORB)
First 30 minutes of trading establish the day's range. Current price vs ORB high/low determines breakout direction and strength.

#### 2.4 News Sentiment (8-K + Bayesian)
Reads latest signals from Supabase `signals` table where `signal_type = '8k_news_bayesian'`. Cross-references SEC 8-K filings with social sentiment.

#### 2.5 Alt Data Signal (V3)
**File:** `scout/scout_alt.py`
Reads asset-specific alternative data signals (CVE for CRWD, FDA for LLY, etc.).

#### 2.6 50% Feature Neutralization (V3)
After computing raw confidence, the Sovereign applies fractional neutralization:
```python
neutralized_conf = raw_conf * (1 - 0.5) + 0.5 * 0.5
```
This removes 50% of the signal's linear exposure to SPY beta, isolating idiosyncratic alpha.

#### 2.7 Score Aggregation
| Signal | Points | Weight |
|--------|--------|--------|
| AFD Momentum | ±2 | Medium |
| Pure Alpha | ±2 | Medium |
| ORB Breakout | ±3 | High |
| News Sentiment | ±2 | Medium |
| **Stockformer** (V3) | **±4** | **Highest** |
| Alt Data (V3) | ±2 | Medium |

Final confidence = `max(0.5, min(neutralized_score, 0.95))`

---

## 3. The Madman Agent

**File:** `bots/madman_agent.py`
**Role:** Tracks retail crowd behavior. Detects FOMO, panic, and exhaustion.

### Signal Stack (V3)

#### 3.1 RSI (Relative Strength Index)
- Overbought: RSI > 70 → fomo += 2
- Oversold: RSI < 30 → fear += 2
- Exuberant: RSI > 85 → triggers Contrarian Fade check

#### 3.2 Contrarian Fade (Highest Priority)
When RSI > 85 (extreme retail FOMO) AND OFI Z-score < -1.5 (institutional absorption):
```
fear += 5  # strong sell signal
```
This detects the exact moment retail is buying into an institutional iceberg order.

#### 3.3 Dynamic Pump Detection (V3)
**File:** `quant/ticker_profiles.py`

Old: flat 5% threshold for all tickers.
New: `pump_threshold = realized_sigma_15min * pump_mult`

| Ticker | Behavior | pump_mult | Effective Threshold |
|--------|----------|-----------|-------------------|
| MSTR | Momentum Breakout | 1.5 | ~4-8% (extreme beta) |
| PLTR | Momentum Breakout | 1.8 | ~3-5% |
| NVDA | Momentum Breakout | 2.0 | ~2-4% |
| JPM | Mean Reversion | 3.0 | ~1-2% |
| NEE | Mean Reversion | 3.0 | ~1-2% |

#### 3.4 Stacked Imbalance Detection (V3)
**File:** `scout/scout_tape.py`

When OFI is positive for 3+ consecutive ticks with average OFI exceeding the 3:1 aggressive/passive ratio threshold → `fomo += 3`. Signals high-conviction directional retail momentum.

#### 3.5 Trapped Order Exhaustion (V3)
When price is within 0.2% of a local 10-tick high AND volume is 2× average AND OFI has flipped below -1.0 Z-score → `fear += 4`. Late retail participants are trapped at the peak.

#### 3.6 Spread Decay Gate (V3)
If the current bid-ask spread exceeds the 90th percentile for this ticker's 9:45 AM window:
```python
confidence *= 0.7  # 30% penalty
```
Prevents execution when friction costs would destroy the 2:1 reward-to-risk ratio.

#### 3.7 Behavior-Adjusted Scoring (V3)
```python
if behavior == "mean_reversion":
    fear = int(fear * 1.3)   # boost fade signals
elif behavior == "momentum_breakout":
    fomo = int(fomo * 1.2)   # boost momentum signals
```

---

## 4. The Scout Layer

### 4.1 Scout News (`scout/scout_news.py`)
- Scrapes SEC EDGAR 8-K filings via `edgartools`
- Scrapes Yahoo Finance / StockTitan headlines
- **Bayesian Adversarial Filter:** If ≥10 social posts share identical phrasing but no 8-K filing exists → classified as bot-farm attack → sentiment nullified

### 4.2 Scout Tape V3 (`scout/scout_tape.py`)
Real-time Alpaca WebSocket streaming for all 10 tickers.

**Event-Based OFI Formula:**
```
e_n = I{P_bid ≥ P_bid_prev} · q_bid
    - I{P_bid ≤ P_bid_prev} · q_bid_prev
    - I{P_ask ≤ P_ask_prev} · q_ask
    + I{P_ask ≥ P_ask_prev} · q_ask_prev
```

This captures the full microstructure of order book changes, not just size imbalance.

**V3 Additions:**
- PyArrow in-memory buffers (5000 rows/symbol, zero-copy columnar)
- Spread tracking per symbol (rolling 60-tick window)
- Stacked Imbalance detection
- Trapped Exhaustion detection
- Previous BBO state tracking for event-based OFI

### 4.3 Scout Alt Data V3 (`scout/scout_alt.py`)
Asset-specific alternative data scraping:

| Ticker | Source | URL | Keywords |
|--------|--------|-----|----------|
| CRWD | NVD CVE Database | services.nvd.nist.gov | zero-day, critical, patch, breach |
| LLY | FDA Press Releases | fda.gov/drugs | Phase 3, fast-track, approval, GLP-1 |
| NVDA | TSMC Supply Chain | Yahoo Finance TSM | CoWoS, HBM, wafer yield, capacity |
| PLTR | Government Contracts | sam.gov | contract, DoD, Army, AIP, Gotham |
| MSTR | Bitcoin Treasury | Yahoo Finance MSTR | BTC purchase, Saylor, convertible |

Each source has a **weight multiplier** (1.0-4.0) reflecting historical alpha correlation. FDA events (weight=4.0) cause the most violent moves.

### 4.4 Scout 13F V3 (`scout/scout_13f.py`)
Quarterly SEC 13F-HR filing scraper for:
- **Renaissance Technologies** (CIK: 0001037389)
- **Citadel Advisors** (CIK: 0001423053)
- **Two Sigma Investments** (CIK: 0001179392)
- **Millennium Management** (CIK: 0001273087)

Parses position changes (accumulation/distribution/exit/new) for watchlist tickers. If Renaissance reduced NVDA by 85%, the Sovereign agent's NVDA confidence gets a 0.7× multiplier.

### 4.5 Crawl News (`scout/crawl_news.py`)
Legacy headline scraper with expanded blocklist (EDGAR, HTTPS, FOMC filtered out). Feeds into Bayesian filter.

---

## 5. The Quant Layer

### 5.1 Feature Factory (`quant/feature_factory.py`)
Two core outputs:
1. **AFD Momentum:** Fractionally differenced price series preserving trend memory
2. **Pure Alpha:** SPY/QQQ-neutralized returns via OLS regression residuals

### 5.2 Regime HMM V3 (`quant/regime_hmm.py`)

**4-State Gaussian Hidden Markov Model:**
| State | Vol Level | Momentum | Trading Behavior |
|-------|-----------|----------|-----------------|
| Chop | Lowest | Near zero | VWAP fade, tight stops (1%/2%) |
| Trend-Bull | Medium | Positive | Full Stockformer, wide stops (2.5%/5%) |
| Trend-Bear | Medium | Negative | Short-only, SQQQ hedge, tight stops (1.5%/3%) |
| Crisis | Highest | Any | Halt all longs, deploy SQQQ + weakest short |

**4 Emission Features:**
1. Realized volatility (20-day, annualized)
2. 5-day momentum
3. VIX proxy (5-day vol / 20-day vol — >1 means vol expanding)
4. Momentum acceleration (5-day mom minus 20-day mom)

**State Assignment:** After fitting, states are labeled by sorting emission means — lowest vol = Chop, highest vol = Crisis, middle two split by momentum sign.

**Backward Compatibility:** V3 states map to V2 names for the engine: `trend_bull → trend`, `trend_bear → trend`.

### 5.3 Ticker Profiles (`quant/ticker_profiles.py`)
Per-symbol behavioral configuration:
- `behavior`: "momentum_breakout", "mean_reversion", or "event_driven"
- `sigma_method`: "implied" (ATM straddle proxy) or "historical"
- `pump_mult`: multiplier on σ for pump threshold
- `spread_pct_90`: 90th percentile bid-ask spread at 9:45 AM
- `sector`: for correlation guard
- `alt_data_sources`: list of high-alpha data sources
- `alt_keywords`: trigger keywords

### 5.4 Meta-Model Ensemble (`quant/meta_model.py`)
Combines three signal streams:
```
final_score = 0.45 × Stockformer + 0.30 × OFI_regression + 0.25 × HMM_posterior
```

**OFI Regression Score:** Converts OFI Z-score + iceberg + stacked + trapped into [-1, 1]:
- Base: `clip(ofi_z × 0.2, -0.5, 0.5)`
- Iceberg: -0.3
- Stacked: +0.25
- Trapped: -0.35

**HMM Score:** Maps regime + confidence to [-1, 1]:
- Crisis: `-1.0 × confidence`
- Trend-Bear: `-0.5 × confidence`
- Trend-Bull: `+0.5 × confidence`
- Chop: `0.0`

**50% SPY Neutralization:** `neutralized = raw - 0.50 × spy_beta_exposure`

**Weight Learning:** Ridge regression on last 30 days of Triple Barrier labeled outcomes (requires TBL labels to be populated first).

### 5.5 Triple Barrier Labeler (`quant/labeler.py`)
Runs nightly on all unlabeled trades in Supabase:

| Barrier | Threshold | Label | Meaning |
|---------|-----------|-------|---------|
| Upper | +4% | 1 | Win (take profit hit first) |
| Lower | -2% | -1 | Loss (stop loss hit first) |
| Vertical | 3:45 PM ET | 0 | Time stop (neither barrier hit) |

Also computes:
- **MFE** (Maximum Favorable Excursion): best unrealized P&L during trade
- **MAE** (Maximum Adverse Excursion): worst unrealized P&L during trade
- **Bars to exit:** how many 1-min bars until barrier was hit

Labels are used as training targets for Stockformer weekly retraining.

### 5.6 Correlation Guard (`quant/correlation_guard.py`)
Prevents sector over-concentration: max 2 positions in the same sector group.

### 5.7 Earnings Filter (`quant/earnings_filter.py`)
Scrapes Yahoo Finance earnings calendar. Skips any symbol with earnings within 48 hours.

---

## 6. The Referee Engine

**File:** `referee/engine_v2.py`
**Cycle:** Every 15 minutes during market hours (4 AM - 8 PM ET).

### 6.1 Cycle Flow
```
1. Check market open
2. Fetch regime (HMM → Supabase → get_latest_regime_full())
3. If Crisis: close longs → deploy SQQQ hedge + weakest short → return
4. If exiting Crisis: close all hedges
5. Fetch account equity
6. Filter watchlist by earnings (48h block)
7. Run Sovereign agent on safe symbols
8. Run Madman agent on safe symbols
9. Run position exit checks
10. For each symbol: compute referee verdict
11. Filter by correlation guard
12. Execute approved trades (Limit IOC)
13. Log ensemble meta-model scores
14. Print cycle summary
```

### 6.2 Verdict Logic
```python
if sovereign == neutral AND madman == neutral → NO_SIGNAL
if sovereign confidence < 75% → NO_SIGNAL
if sovereign == madman (same direction) → CROWDED_SKIP
if sovereign ≠ madman (opposing) AND sovereign ≥ 75% → EXECUTE (STRONG)
if madman == neutral AND sovereign ≥ 80% → EXECUTE (WEAK)
```

### 6.3 Crisis Hedging (V3)
When HMM Crisis confidence ≥ 90%:
1. **SQQQ Buy:** 3% equity into 3× inverse QQQ ETF (Limit IOC)
2. **Weakest Short:** 2% equity shorting the watchlist symbol with worst 5-day return (Limit DAY)

Position manager preserves both during force-close cycles.

### 6.4 Regime-Specific Stops (V3)
| Regime | Stop Loss | Take Profit |
|--------|-----------|-------------|
| Trend (V2 compat) | 2.0% | 4.0% |
| Trend-Bull (V3) | 2.5% | 5.0% |
| Trend-Bear (V3) | 1.5% | 3.0% |
| Chop | 1.0% | 2.0% |
| Crisis | N/A | N/A |

### 6.5 Execution
All orders are **Limit IOC** (Immediate Or Cancel):
- Buy: `limit_price = mid × 1.0005` (0.05% above mid)
- Sell: `limit_price = mid × 0.9995` (0.05% below mid)
- Whole shares only (fractional not supported for Limit IOC)

### 6.6 Position Sizing
```
risk_pct = min(BASE_RISK(2%) + kelly_boost, MAX_RISK(5%)) × size_mult
qty = int(equity × risk_pct / mid_price)
```
- Strong signal: `size_mult = 1.0`
- Weak signal: `size_mult = 0.6`

---

## 7. Position Management & Risk Controls

**File:** `referee/position_manager.py`

### Exit Triggers (in priority order)
1. **Crisis Force-Close:** Close all longs (preserves SQQQ hedge + crisis shorts)
2. **EOD Close:** 15 min before market close → close everything
3. **Stop Loss / Take Profit:** Regime-adjusted thresholds
4. **Signal-Based Exit:** If both bots flip against the position

### Risk Limits
- Max 4 open positions simultaneously
- Max 2 positions per sector (correlation guard)
- Max 5% equity per position
- 48-hour earnings blackout
- Spread decay gate (Madman confidence × 0.7 if spread > 90th percentile)

---

## 8. Infrastructure & Deployment

### 8.1 launchd Services
| Service | Plist | Interval | What it does |
|---------|-------|----------|-------------|
| Trader | `com.tugofwar.trader.plist` | 15 min cycles | Runs `referee/engine_v2.py` |
| Tape | `com.tugofwar.tape.plist` | Continuous | Runs `scout/scout_tape.py` WebSocket stream |

### 8.2 Background Schedulers (inside engine_v2.py)
| Task | Interval | What it does |
|------|----------|-------------|
| `run_cycle()` | 15 min | Main trading cycle |
| `_run_scout_background()` | 30 min | Runs scout_news.py + scout_alt.py |
| `_run_hmm_background()` | 60 min | Re-fits HMM and logs regime |

### 8.3 Dashboard
**Directory:** `dashboard/`
**Stack:** Next.js + Supabase real-time + TailwindCSS

Components:
- `TugMeter.tsx` — visual tug-of-war gauge showing sovereign vs madman tension
- `TradeRow.tsx` — trade history with implementation shortfall column
- Regime badge (color-coded: green=trend, yellow=chop, red=crisis)
- OFI Z-score display per ticker
- Iceberg indicator

### 8.4 M1 Optimizations
- PyTorch MPS backend for Stockformer training/inference
- PyArrow zero-copy columnar buffers for OFI data
- hmmlearn with sklearn StandardScaler (ARM-native via Accelerate framework)
- All heavy compute in background threads, never blocking the 15-min cycle

---

## 9. Database Schema

### Core Tables (V2)
- `trades` — all executed orders with Alpaca IDs, shortfall, status
- `signals` — all bot signals (sovereign, madman, scout_news, scout_alt, scout_13f)
- `tug_results` — referee verdict per symbol per cycle
- `ofi_snapshots` — OFI Z-scores, iceberg flags, per symbol per minute
- `regime_states` — HMM regime state, confidence, volatility, momentum

### V3 Additions
- `trades.tbl_label` — Triple Barrier label (1=win, -1=loss, 0=time_stop)
- `trades.stockformer_score` — Stockformer conviction at time of trade
- `trades.ensemble_score` — Meta-model ensemble score at time of trade
- `signals.stacked_imbalance` — Stacked Imbalance flag
- `signals.trapped_exhaustion` — Trapped Exhaustion flag
- `regime_states.state_v3` — V3 state name (chop/trend_bull/trend_bear/crisis)
- `regime_states.vix_proxy` — Short-term vol / long-term vol ratio

---

## 10. Mathematical Foundations

### 10.1 Fractional Differencing
```
(1 - B)^d X_t = Σ_{k=0}^{∞} (-1)^k C(d,k) B^k X_t

Weight recursion: w_k = -w_{k-1} × (d - k + 1) / k
Optimal d: min d ∈ [0,1] s.t. ADF p-value < 0.05
```

### 10.2 Event-Based OFI
```
e_n = I{P^B_n ≥ P^B_{n-1}} × q^B_n
    - I{P^B_n ≤ P^B_{n-1}} × q^B_{n-1}
    - I{P^A_n ≤ P^A_{n-1}} × q^A_n
    + I{P^A_n ≥ P^A_{n-1}} × q^A_{n-1}

Price impact: ΔP = β × OFI
```

### 10.3 Kelly Criterion
```
f* = (b × p - q) / b
where: b = avg_win/avg_loss, p = win_rate, q = 1 - p
Applied at 50% Kelly (half-Kelly for safety)
Capped at 25% max allocation
```

### 10.4 Sharpe Loss (Stockformer)
```
L = -(mean(portfolio_returns) / std(portfolio_returns))
portfolio_returns = Σ(prediction_i × actual_return_i)
```

### 10.5 Feature Neutralization
```
Y_neutral = Y - F(F^T F)^{-1} F^T Y
Applied at 50% proportion: Y_final = Y × 0.5 + Y_neutral × 0.5
```

### 10.6 HMM Forward Algorithm
```
P(state_t | observations_{1:t}) = α_t(state) / Σ_s α_t(s)
α_t(s) = emission(obs_t | s) × Σ_{s'} transition(s' → s) × α_{t-1}(s')
```

---

## 11. Supabase Schema Migration (V3)

**IMPORTANT:** Copy and paste the following SQL into your Supabase SQL Editor (Dashboard → SQL Editor → New Query):

```sql
-- V3 Schema Migration
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tbl_label INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS stockformer_score REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ofi_regression_score REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS hmm_posterior REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ensemble_score REAL;

ALTER TABLE signals ADD COLUMN IF NOT EXISTS ticker_personality TEXT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spread_percentile REAL;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS stacked_imbalance BOOLEAN DEFAULT FALSE;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS trapped_exhaustion BOOLEAN DEFAULT FALSE;

ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS state_v3 TEXT;
ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS vix_proxy REAL;
ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS momentum_accel REAL;
```

After running this SQL, the V3 features will be fully active. Until then, the system operates in V2-compatible mode (all V3 columns are optional).

---

## File Tree
```
Tug-Of-War System/
├── bots/
│   ├── sovereign_agent.py    # Institutional bot (AFD + Stockformer + ORB + news + alt data)
│   └── madman_agent.py       # Retail bot (RSI + OFI + pump + stacked + trapped + personality)
├── quant/
│   ├── feature_factory.py    # AFD + SPY/QQQ neutralization
│   ├── stockformer.py        # Multi-asset Transformer (MPS GPU)
│   ├── regime_hmm.py         # 4-state HMM (Chop/Bull/Bear/Crisis)
│   ├── meta_model.py         # Ensemble scorer (Ridge-weighted)
│   ├── ticker_profiles.py    # Per-ticker personality configs
│   ├── labeler.py            # Triple Barrier nightly labeler
│   ├── backtester.py         # Historical backtester
│   ├── earnings_filter.py    # 48-hour earnings blackout
│   └── correlation_guard.py  # Max 2 per sector
├── referee/
│   ├── engine_v2.py          # Main decision engine (15-min cycles)
│   └── position_manager.py   # Exit logic + crisis hedge preservation
├── scout/
│   ├── scout_tape.py         # Real-time OFI WebSocket (event-based)
│   ├── scout_news.py         # 8-K + Bayesian adversarial filter
│   ├── scout_alt.py          # CVE/FDA/TSMC/gov alt data
│   ├── scout_13f.py          # 13F institutional telemetry
│   └── crawl_news.py         # Legacy headline scraper
├── dashboard/                # Next.js real-time dashboard
├── db/
│   ├── schema.sql            # V2 base schema
│   └── schema_v3.sql         # V3 additions
├── models/
│   └── stockformer.pt        # Trained Stockformer weights (579KB)
├── docs/
│   └── V3_ARCHITECTURE.md    # This file
├── logs/                     # Engine + tape logs
├── .env                      # API keys (gitignored)
├── .env.example              # Template
├── .gitignore
├── README.md
└── requirements.txt
```
