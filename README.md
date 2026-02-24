# Tug-Of-War Sovereign Quant Trading System

> An asynchronous, event-driven quantitative trading engine that sits at 0% CPU, consumes real-time market microstructure data across three independent scout processes, and strikes in milliseconds when the math aligns. Built entirely in Python on macOS — no Docker, no cloud VMs, no heavy frameworks.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Alpaca](https://img.shields.io/badge/broker-Alpaca-yellow.svg)](https://alpaca.markets/)
[![Supabase](https://img.shields.io/badge/database-Supabase-green.svg)](https://supabase.com/)
[![ZeroMQ](https://img.shields.io/badge/messaging-ZeroMQ%20TCP-red.svg)](https://zeromq.org/)
[![Next.js](https://img.shields.io/badge/dashboard-Next.js-black.svg)](https://nextjs.org/)

---

## How Data Flows: Exchange → Broker

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                    PHASE 1: SCOUTS                       │
                        │                                                         │
                        │  scout_tape.py ──── OFI, Iceberg, Trapped Exhaustion    │
                        │  scout_dix.py  ──── Dark Pool Short Volume, DIX Index   │
                        │  regime_hmm.py ──── GEX, SPY Vol → 4-State HMM         │
                        │                          │                              │
                        │                    ZeroMQ TCP                            │
                        │               tcp://127.0.0.1:5555                      │
                        └──────────────────────┬──────────────────────────────────┘
                                               │ JSON payload (<1ms)
                        ┌──────────────────────▼──────────────────────────────────┐
                        │                 PHASE 2: ALPHA FACTORY                   │
                        │                                                         │
                        │  Stockformer ──── Deep-learning momentum (neutralized)  │
                        │  Alpha Factory ── 19 WorldQuant 101 formulaic alphas    │
                        │  Cross-Sectional Deviation ── vs. watchlist median      │
                        │  4-Stream Ensemble ── Ridge regression weights          │
                        │  Zero-Sum Normalization ── beta-neutral portfolio       │
                        └──────────────────────┬──────────────────────────────────┘
                                               │ alpha signals μ[20]
                        ┌──────────────────────▼──────────────────────────────────┐
                        │            PHASE 3: QUANTUM ALLOCATOR                    │
                        │                                                         │
                        │  E(w) = -(w'μ) + λ(w'Σw) + c∑|Δw|                     │
                        │  Simulated Annealing · 5000 iterations · ~135ms         │
                        │  Metropolis tunneling · regime-scaled risk aversion     │
                        │  Output: optimal weight array → integer shares          │
                        └──────────────────────┬──────────────────────────────────┘
                                               │ target portfolio
                        ┌──────────────────────▼──────────────────────────────────┐
                        │               PHASE 4: EXECUTION                         │
                        │                                                         │
                        │  Turnover Guard ──── 12bps hurdle to flip positions     │
                        │  Maker/Taker Router ── passive in Chop, aggressive in   │
                        │                        Trend (micro-price + 2¢)         │
                        │  LOB Queue Simulator ── paper-trade realism penalty     │
                        │  ATR Brackets ──── dynamic stop/take on every fill      │
                        │  TCA Logger ──── market impact + adverse selection      │
                        └──────────────────────┬──────────────────────────────────┘
                                               │
                                          ALPACA API
```

---

## The Core Idea

Most trading bots follow indicators blindly. This system watches **two opposing forces** — institutional logic vs. retail sentiment — and only trades when they *disagree*. That disagreement is the edge.

The engine then passes signals through a **Simulated Annealing portfolio optimizer** that finds the mathematically perfect allocation across 20 symbols, accounting for live correlations and transaction costs. Finally, a **microstructure-aware execution layer** routes orders based on the current market regime to minimize slippage.

---

## System Architecture

```
Tug-Of-War System/
│
├── scout/                         # Phase 1: The Scouts (3 independent processes)
│   ├── scout_tape.py              # Lit Tape — L1/L2 OFI, iceberg, trapped exhaustion
│   ├── scout_dix.py               # Dark Tape — FINRA short volume, dark pool DIX
│   ├── scout_news.py              # SEC 8-K + Yahoo/StockTwits + Bayesian bot-farm filter
│   ├── scout_alt.py               # Alternative data (FDA, NVD CVE, patent filings)
│   ├── scout_13f.py               # 13-F institutional holdings tracker
│   └── sources.py                 # Data source registry
│
├── quant/                         # Phase 2: The Alpha Factory
│   ├── stockformer.py             # Transformer model — temporal + cross-asset attention
│   ├── alpha_factory.py           # 19 WorldQuant 101 formulaic alphas
│   ├── feature_factory.py         # AFD + SPY/QQQ neutralization + cross-sectional deviation
│   ├── meta_model.py              # 4-stream Ridge ensemble (SF + OFI + HMM + Alpha)
│   ├── quantum_allocator.py       # Simulated Annealing portfolio optimizer
│   ├── regime_hmm.py              # 4-state Gaussian HMM + GEX post-hoc adjustment
│   ├── tca.py                     # Transaction Cost Analysis (impact + adverse selection)
│   ├── synthetic_market.py        # GBM + TimeGAN stress testing
│   ├── labeler.py                 # Triple Barrier with ATR-dynamic thresholds
│   ├── correlation_guard.py       # Sector over-concentration filter
│   ├── earnings_filter.py         # 48-hour earnings blackout
│   ├── ticker_profiles.py         # Per-symbol personality profiles
│   └── backtester.py              # Historical replay engine
│
├── bots/                          # Phase 2: The Agents
│   ├── sovereign_agent.py         # Institutional: AFD + ORB + VWAP + 8-K + Half-Kelly
│   └── madman_agent.py            # Retail/Contrarian: RSI + OFI fade + volume spike
│
├── referee/                       # Phase 3 & 4: Orchestration + Execution
│   ├── engine_v2.py               # Core engine — event-driven, ZMQ poll, quantum allocator
│   ├── event_bus.py               # ZeroMQ TCP Pub/Sub (brokerless, <100ms latency)
│   ├── position_manager.py        # Auto-exit: stop, take, flip, crowded, EOD, crisis
│   ├── net_guard.py               # Network resilience (retry decorator, offline cache)
│   └── secret_vault.py            # macOS Keychain secret management
│
├── dashboard/                     # Next.js live dashboard (localhost:3000)
│   ├── app/page.tsx               # Regime badge, tug meters, trade history
│   └── app/components/            # TugMeter, TradeRow, etc.
│
├── db/schema.sql                  # Supabase table definitions + RLS policies
├── com.tugofwar.trader.plist      # macOS launchd: engine runs 24/7
├── com.tugofwar.tape.plist        # macOS launchd: OFI tape streams 24/7
├── requirements.txt               # Python dependencies
└── .env.example                   # Environment variable template
```

---

## Phase 1: The Scouts

Three independent Python processes run simultaneously, watching the physical mechanics of the market. They do not wait — the millisecond a signal is detected, a JSON payload fires through ZeroMQ.

### The Lit Tape (`scout_tape.py`)

Watches public exchanges for real-time microstructure events:

| Signal | Detection |
|---|---|
| **Order Flow Imbalance** | `OFI = (bid_size − ask_size) / (bid_size + ask_size)`, Z-scored over 60-tick window |
| **Trapped Exhaustion** | OFI flip at breakout peak + volume spike — retail trapped at the top |
| **Institutional Iceberg** | Price flat + negative OFI — smart money absorbing retail buy pressure |
| **Stacked Imbalance** | 3:1 aggressive/passive ratio across 3+ ticks — momentum signal |
| **Spread Blow** | Spread widens >3x average — liquidity withdrawal, market makers pulling quotes |

### The Dark Tape (`scout_dix.py`)

Scrapes FINRA short volume and dark pool prints to see where whales are quietly accumulating:

| Signal | Detection |
|---|---|
| **Dark Accumulation** | Short ratio declining over 3+ days + volume above 1.5x average |
| **Dark Distribution** | Short ratio spiking over 3+ days + volume above 1.5x average |
| **DIX Extreme** | Aggregate Dark Pool Index at extreme levels (>55% = bullish, <35% = bearish) |

### The Macro State (`regime_hmm.py`)

4-state Gaussian HMM trained on SPY daily returns, volatility, momentum, and VIX proxy:

| Regime | Behavior | Risk Aversion (λ) |
|---|---|---|
| **TREND_BULL** | Full momentum mode — wide stops, aggressive sizing | 1.0 |
| **TREND_BEAR** | Cautious trend-following — tighter stops | 2.0 |
| **CHOP** | Mean-reversion scalps only — ORB entries, passive execution | 3.0 |
| **CRISIS** | All trading halted, positions force-closed, SQQQ hedge deployed | 5.0 |

**GEX Post-Hoc Adjustment**: Options Gamma Exposure modulates the HMM output — positive GEX biases toward Chop (mean-reverting), negative GEX biases toward Crisis (volatility explosion).

### ZeroMQ Event Bus (`event_bus.py`)

```
Transport:  tcp://127.0.0.1:5555 (brokerless, zero-config)
Pattern:    PUB/SUB with topic-based filtering
Latency:    <100ms end-to-end (scout → engine reaction)
Fallback:   If ZMQ unavailable, events silently dropped (polling still works)
```

---

## Phase 2: The Alpha Factory

The engine rests at **0% CPU**. The moment it receives a ZMQ signal or a scheduled cycle fires, it wakes up and computes alpha across four independent streams.

### Stream 1: Stockformer — Deep Learning Conviction (`stockformer.py`)

A custom Transformer with temporal attention (across time) and cross-asset attention (across the 20-symbol watchlist). Trained on a negative Sharpe ratio loss function.

**Numerai Feature Neutralization**: After the model produces raw scores, they are regressed against input features (AFD-close, volume, pure_alpha, realized_vol) and the linear component is subtracted. This forces the Stockformer to only output **non-linear alpha** — the hidden relationships that simple momentum can't explain.

### Stream 2: OFI Microstructure Score (`meta_model.py`)

Converts real-time Order Flow Imbalance, iceberg detection, stacked imbalance, and trapped exhaustion into a single regression-like score in [-1, 1].

### Stream 3: HMM Regime Posterior (`meta_model.py`)

Maps the current HMM state and confidence into a directional score. Crisis = strong sell, Trend_Bull = mild buy, Chop = neutral.

### Stream 4: WorldQuant Alpha Factory (`alpha_factory.py`)

19 formulaic micro-alphas from Kakushadze's "101 Formulaic Alphas" paper, including:

| Alpha | Formula | What It Captures |
|---|---|---|
| `#1` | `ts_argmax(signed_power(returns, 2), 5)` | Recent momentum concentration |
| `#12` | `sign(Δvolume) × (-Δclose)` | Volume-price divergence |
| `#17` | `-rank(ts_rank(close,10)) × rank(Δ²close) × rank(ts_rank(vol/adv20,5))` | Triple-ranked momentum |
| `#41` | `√(high × low) - vwap` | Geometric mean vs. institutional anchor |
| `#49` | Momentum regime switch | Acceleration/deceleration detection |
| `#53` | `-Δ((close-low)/(high-low), 9)` | Intrabar position shift |

All 19 alphas are **cross-sectionally ranked** across the watchlist, then combined into a single composite score.

### Cross-Sectional Deviation (Optiver Strategy)

Strips macro-market noise by subtracting the watchlist median from each symbol's feature:

```
NVDA_OFI_Dev = NVDA_OFI − Median(Watchlist_OFI)
```

Only trade NVDA if its order flow is **exceptionally aggressive vs. peers** — not just because the whole market is moving.

### 4-Stream Ensemble

| Stream | Weight | Source |
|---|---|---|
| Stockformer (feature-neutralized) | 0.35 | `stockformer.py` |
| OFI Microstructure (cross-sectional deviated) | 0.25 | `meta_model.py` |
| HMM Regime Posterior | 0.20 | `regime_hmm.py` |
| WorldQuant Alpha Composite | 0.20 | `alpha_factory.py` |

**Zero-Sum Post-Processing**: Ensemble scores are forced to sum to zero across the watchlist, creating a **beta-neutral micro-portfolio** — if buying 3 tech stocks, they auto-hedge against each other.

---

## Phase 3: The Quantum-Inspired Allocator (`quantum_allocator.py`)

Replaces rigid per-symbol Kelly sizing with a **portfolio-level optimizer** that finds the mathematically perfect weight array across all 20 symbols.

### The Objective Function

```
E(w) = -(w'μ) + λ(w'Σw) + c∑|w_i - w_current_i|
```

| Term | Meaning |
|---|---|
| `w'μ` | **Alpha** — expected return of portfolio based on ensemble signals |
| `λ(w'Σw)` | **Risk** — portfolio variance from live covariance matrix (penalizes correlated holdings) |
| `c∑\|Δw\|` | **Friction** — literal dollar cost of crossing the bid-ask spread to rebalance |

### Simulated Annealing Algorithm

1. **Heat up**: Start at high temperature T. Randomly mutate weight allocations. At high T, occasionally accept *worse* portfolios — this **quantum tunneling** prevents getting stuck in local traps.

2. **Metropolis Criterion**: Probability of accepting a worse state:
   ```
   P = exp(-ΔE / T)
   ```

3. **Cool down**: T drops geometrically (×0.9985 per iteration) over 5000 steps. As it cools, the algorithm strictly hones in on the global optimum — the exact fraction of shares to hold that perfectly balances alpha, correlations, and fees.

### Performance

| Metric | Value |
|---|---|
| Iterations | 5,000 |
| Runtime | ~135ms on Apple Silicon |
| Implementation | Pure vectorized NumPy (no scipy, no cvxpy) |
| Max weight | 20% per position |
| Exposure limits | 80% long / 20% short |
| Covariance | Ledoit-Wolf shrinkage (0.3 blend toward diagonal) |

---

## Phase 4: Execution

The engine has the perfect target portfolio. Before firing orders to Alpaca, every trade runs through the **microstructure gauntlet**.

### Maker/Taker Dynamic Routing

Execution routing is tied directly to the HMM regime:

| Regime | Mode | Strategy | Time-in-Force |
|---|---|---|---|
| **Chop** | MAKER | Post passive limit at the bid (buy) or ask (sell). Capture the spread, earn exchange rebates, zero market impact. | DAY |
| **Trend** | TAKER | Aggressive marketable limit at Stoikov micro-price + 2¢. Pay the fee, cross the spread, guarantee fill before the train leaves. | IOC |

**Stoikov Micro-Price** (2018): Weights the midpoint by bid/ask volume imbalance. If `bid_size >> ask_size`, the true price is closer to the ask (buyers are aggressive).

```
P_micro = bid + (bid_size / (bid_size + ask_size)) × (ask - bid)
```

### LOB Queue Position Simulator

Paper trading APIs are liars — they fill your orders the instant the price touches your limit. In reality, you're at the **back of the queue**.

The queue simulator only counts a passive fill if the market traded **through** your limit price:

```
BUY  @ $150.00 → only fills if mid drops to $149.99 (1bp through)
SELL @ $150.00 → only fills if mid rises to $150.01 (1bp through)
```

If the queue sim fails, the order is cancelled and logged as `queue_blocked`. This violently penalizes backtesting and forces the strategy to be robust against missed fills.

### Alpha Decay & Turnover Penalty

With 19 WorldQuant alphas generating frequent signals, high turnover kills micro-funds. The engine blocks position flips unless:

```
E[R_new] > E[R_current] + 2 × (Fees + Slippage)
```

| Parameter | Value |
|---|---|
| Estimated fee per side | 1.0 bps |
| Estimated slippage per side | 2.0 bps |
| Round-trip cost | 6.0 bps |
| Flip hurdle | 12.0 bps (2× round-trip) |

New entries always pass — only flips are gated.

### Transaction Cost Analysis (`tca.py`)

Every fill is measured:

| Metric | What It Tracks |
|---|---|
| **Market Impact** | Signed difference between micro-price and fill price |
| **Adverse Selection** | Post-fill price movement at 1s, 10s, 60s intervals |
| **Opportunity Cost** | Alpha of signals that were skipped (correlation guard, max positions, turnover filter) |

### ATR-Based Dynamic Brackets

Every fill gets server-side stop-loss and take-profit orders that survive laptop sleep:

```
Take Profit = 1.5 × ATR(20)    (2:1 reward/risk ratio)
Stop Loss   = 0.75 × ATR(20)
```

---

## The Agents

### Sovereign Agent — Institutional Logic (`sovereign_agent.py`)

| Signal | Points | Description |
|---|---|---|
| **AFD Momentum** | 3 | Adaptive Fractional Differencing — minimum `d` for stationarity while preserving memory |
| **Pure Alpha** | 2 | SPY + QQQ beta stripped via OLS — isolates ticker-specific momentum |
| **ORB** | 3 | Opening Range Breakout — first 30-min high/low sets the range |
| **VWAP Deviation** | 2 | Price vs. intraday VWAP — institutional anchor |
| **News Sentiment** | 2 | SEC 8-K + Bayesian bot-farm filter |
| **Stockformer** | 4 | Deep-learning conviction (highest weight) |
| **Alt Data** | 2 | FDA filings, CVE databases, patent activity |
| **RSI** | 2 | Oversold/overbought levels |
| **MACD** | 2 | Histogram momentum |
| **Volume Surge** | 2 | Today's volume vs. 20-day average |
| **52-Week Position** | 2 | Near highs = momentum, near lows = mean-reversion |
| **MA20** | 2 | Price vs. 20-day moving average |
| **Intraday Momentum** | 2 | 1-hour price slope from 5-min bars |

**Position Sizing**: Half-Kelly with kurtosis penalty — fat-tailed assets get reduced sizing because they have more extreme adverse moves than a normal distribution predicts.

### Madman Agent — Retail/Contrarian Logic (`madman_agent.py`)

| Signal | Description |
|---|---|
| **RSI Pump** | RSI >70 = retail FOMO, RSI <30 = retail fear |
| **OFI Contrarian Fade** | RSI >85 AND OFI Z <−1.5 = retail exuberant but smart money absorbing → SELL |
| **Trapped Exhaustion** | OFI flip at breakout peak + volume spike — retail trapped |
| **Volume Spike** | Today's volume >2× 20-day average |
| **Institutional Iceberg** | Price flat + negative OFI = smart money absorbing |

---

## Risk Controls

| Control | Rule |
|---|---|
| Max open positions | 4 at any time |
| Base risk per trade | 3% of equity (floor) |
| Max risk per trade | 6% of equity (quantum allocator ceiling) |
| Stop loss | ATR-based dynamic (0.75× ATR) |
| Take profit | ATR-based dynamic (1.5× ATR) |
| Earnings filter | Skip any symbol with earnings within 48 hours |
| Correlation guard | Max 2 positions from same sector group |
| Turnover filter | 12bps hurdle to flip positions |
| Queue simulator | Passive fills require price to trade through limit |
| Crisis streak | Require 2 consecutive crisis readings before acting |
| Crisis hedge | SQQQ (3× inverse QQQ) + weakest-symbol short |
| EOD close | All positions closed by 3:45 PM ET |
| High-noise window | No new entries 9:30-10:00 and 15:45-16:00 ET |
| Offline resilience | Server-side brackets survive laptop sleep; cached regime for offline cycles |

---

## Watchlist

**Trading targets (20):**

```
NVDA · CRWD · LLY · TSMC · JPM · NEE · CAT · SONY · PLTR · MSTR
MSFT · AMZN · META · GLD · XLE · UBER · AMD · COIN · MRNA · IWM
```

**Silent benchmarks (not traded):** `SPY · QQQ` — used for regime detection, feature neutralization, and covariance estimation.

---

## Event-Driven Main Loop

The engine does **not** poll on a timer. It uses ZeroMQ's `zmq.Poller` to sleep at the OS kernel level:

```
┌─────────────────────────────────────────────────┐
│              zmq.Poller (epoll/kqueue)           │
│                                                  │
│  Wakes on:                                       │
│    (a) ZMQ event from scout_tape.py  → <100ms    │
│    (b) 1-second timeout              → check     │
│        schedule.run_pending()                    │
│                                                  │
│  CPU when idle: 0%                               │
│  CPU during cycle: brief spike, then back to 0%  │
└─────────────────────────────────────────────────┘
```

---

## Database Schema (Supabase)

| Table | Purpose |
|---|---|
| `signals` | Every signal from both bots — regime state, OFI Z-score, AFD momentum, Bayesian score |
| `tug_results` | Every referee decision — verdict, tug score, conflict flag |
| `trades` | Every order — side, qty, limit price, shortfall bps, maker/taker mode, queue status |
| `regime_states` | HMM state history — 4-state with confidence + SPY vol/momentum + GEX |
| `ofi_snapshots` | Per-symbol OFI Z-score + iceberg flag, written every 60 seconds |
| `tca_metrics` | Market impact, adverse selection (1s/10s/60s), opportunity cost |
| `dark_pool_signals` | Dark pool accumulation/distribution signals + aggregate DIX |

---

## 24/7 Schedule

| Interval | Task |
|---|---|
| **Continuous** | ZMQ poll — 0% CPU idle, wakes on scout events |
| **Every 15 min** | Full trading cycle (alpha factory → allocator → execution) |
| **Every 30 min** | Scout news + alt data + dark pool refresh |
| **Every 60 min** | HMM regime re-fit |
| **Every 60 sec** | OFI tape flush to Supabase |
| **Every 4 hours** | Triple Barrier labeler grades outcomes, Ridge weights update |
| **Weekly (Sunday)** | Stockformer retrain on 180 days of daily bars |
| **3:45 PM ET** | EOD close — all positions exited |

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/HectorSalomanca/Nettss-Tug-Of-War-Trading-bot.git
cd Nettss-Tug-Of-War-Trading-bot
pip3 install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in your keys
```

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADE=True
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key
USER_ID=your-supabase-user-uuid
```

Get Alpaca keys at [alpaca.markets](https://alpaca.markets/) (free paper trading).
Get Supabase keys at [supabase.com](https://supabase.com/) (free tier).

### 3. Initialize Database

Run `db/schema.sql` in your Supabase SQL editor.

### 4. Run

```bash
# Single test cycle
python3 referee/engine_v2.py --once

# 24/7 event-driven (recommended)
python3 referee/engine_v2.py --interval 15

# OFI tape streamer (separate terminal)
python3 scout/scout_tape.py

# Dark pool scanner (standalone test)
python3 scout/scout_dix.py

# Quantum allocator self-test
python3 quant/quantum_allocator.py

# Alpha factory self-test
python3 quant/alpha_factory.py

# Backtester
python3 quant/backtester.py --symbols NVDA PLTR JPM --days 365
```

### 5. Dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### 6. 24/7 macOS Background Service

```bash
cp com.tugofwar.trader.plist ~/Library/LaunchAgents/
cp com.tugofwar.tape.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tugofwar.trader.plist
launchctl load ~/Library/LaunchAgents/com.tugofwar.tape.plist

# Monitor logs
tail -f logs/engine.log
tail -f logs/tape.log
```

---

## What Makes This Different

| Feature | Basic Bot | This System |
|---|---|---|
| Architecture | Polling loop | Event-driven (ZMQ poll, 0% CPU idle) |
| Signal source | Single indicator | 4-stream ensemble (Stockformer + OFI + HMM + 19 Alphas) |
| Position sizing | Fixed dollar or Kelly | Quantum-inspired Simulated Annealing portfolio optimizer |
| Market regime | Ignored | 4-state HMM with GEX adjustment — strategy adapts automatically |
| Feature engineering | Raw prices | Cross-sectional deviation + feature neutralization (Optiver/Numerai) |
| Alpha generation | One model | 19 WorldQuant 101 formulaic alphas, cross-sectionally ranked |
| Execution | Market orders | Maker/Taker routing by regime + Stoikov micro-price |
| Paper realism | Instant fills | LOB queue position simulator — price must trade through limit |
| Turnover control | None | 12bps hurdle to flip (E[R_new] > E[R_current] + 2×costs) |
| Dark pools | Invisible | FINRA short volume + DIX aggregate tracking |
| Risk management | Stop loss only | ATR brackets + 7 exit rules + crisis hedge + correlation guard |
| Microstructure | None | Real-time OFI + iceberg + trapped exhaustion + spread blow |
| Cost analysis | None | Full TCA: market impact, adverse selection, opportunity cost |
| Secrets | `.env` file | macOS Keychain encryption |

---

## Python Dependencies

```
alpaca-py>=0.20.0       # Alpaca trading + data API
supabase>=2.0.0         # Database client
python-dotenv>=1.0.0    # Environment variables
pandas>=2.0.0           # Data manipulation
numpy>=1.26.0           # Numerical computing (allocator, alphas)
hmmlearn>=0.3.0         # Gaussian HMM for regime detection
statsmodels>=0.13.0     # ADF stationarity test (for AFD)
fracdiff>=0.9.0         # Fractional differencing
scikit-learn>=1.3.0     # Feature scaling, Ridge regression
edgartools>=2.0.0       # SEC EDGAR 8-K filing access
httpx>=0.27.0           # Async HTTP for news scraping
beautifulsoup4>=4.12.0  # HTML parsing
schedule>=1.2.0         # Cycle scheduling
pytz>=2024.1            # Market hours timezone handling
pyzmq>=25.0.0           # ZeroMQ event bus (TCP Pub/Sub)
torch>=2.0.0            # Stockformer + TimeGAN (optional, MPS accelerated)
```

---

## Disclaimer

This system trades on a **paper (simulated) account only**. Built for educational and research purposes. Past backtest performance does not guarantee future results. Do not use with real money without extensive additional testing.

---

## License

MIT License
