"""
Tug-Of-War Backtester

Replays 1 year of daily OHLCV data through the same logic as the live engine:
  - Sovereign signals: AFD momentum + SMA crossover + Pure Alpha
  - Madman signals: RSI + volume spike
  - Referee verdict: conflict filter (house-edge model)
  - Position sizing: Kelly Criterion (2-5% equity)
  - Exit rules: stop loss, take profit, signal flip, EOD close

Outputs:
  - Win rate, total return, Sharpe ratio, max drawdown
  - Per-trade log CSV
  - Equity curve printed to terminal

Usage:
  python3 quant/backtester.py
  python3 quant/backtester.py --symbols NVDA PLTR JPM --days 365
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.feature_factory import compute_afd_momentum, neutralize, compute_returns

from referee.secret_vault import get_secret

ALPACA_API_KEY    = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

DEFAULT_SYMBOLS = ["NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR"]
INITIAL_EQUITY  = 100_000.0
BASE_RISK_PCT   = 0.02
MAX_RISK_PCT    = 0.05
STOP_LOSS_PCT   = 0.02
TAKE_PROFIT_PCT = 0.04
MAX_OPEN_TRADES = 4
SOVEREIGN_MIN_CONF  = 0.80
SOVEREIGN_SOLO_CONF = 0.85
MIN_HOLD_BARS   = 3
BUY_ONLY        = True       # paper account: no shorting


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_history(symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df   = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.sort_index().reset_index()
    except Exception as e:
        print(f"[BACKTEST] Fetch error {symbol}: {e}")
        return None


# ── Signal Generation (vectorised over history) ───────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.full(len(closes), 50.0)
    for i in range(period + 1, len(closes)):
        deltas = np.diff(closes[i - period - 1: i])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag = np.mean(gains)
        al = np.mean(losses)
        if al == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100 - (100 / (1 + ag / al))
    return rsi


def sovereign_signal(closes: np.ndarray, spy_closes: np.ndarray,
                     qqq_closes: np.ndarray, i: int) -> tuple:
    """Returns (direction, confidence) at bar i using data up to i."""
    if i < 55:
        return "neutral", 0.5

    window = closes[:i+1]
    spy_w  = spy_closes[:i+1]
    qqq_w  = qqq_closes[:i+1]

    sma20 = np.mean(window[-20:])
    sma50 = np.mean(window[-50:]) if len(window) >= 50 else sma20
    mom5  = (window[-1] - window[-6]) / window[-6] if len(window) >= 6 else 0

    # Pure alpha (neutralized momentum)
    residuals = neutralize(window, spy_w, qqq_w)
    pure_alpha = float(np.mean(residuals[-5:])) if len(residuals) >= 5 else 0.0

    # AFD momentum
    try:
        afd_mom = compute_afd_momentum(window)
    except Exception:
        afd_mom = 0.0

    bull = bear = 0
    if window[-1] > sma20: bull += 1
    else: bear += 1
    if sma20 > sma50: bull += 2
    else: bear += 2
    if mom5 > 0.02: bull += 1
    elif mom5 < -0.02: bear += 1
    # AFD and pure alpha weighted 3x — core quant edge
    if afd_mom > 0.0001: bull += 3
    elif afd_mom < -0.0001: bear += 3
    if pure_alpha > 0.0005: bull += 3
    elif pure_alpha < -0.0005: bear += 3

    total = bull + bear
    if total == 0:
        return "neutral", 0.5
    if bull > bear:
        return "buy", round(min(0.5 + (bull / total) * 0.45, 0.95), 4)
    return "sell", round(min(0.5 + (bear / total) * 0.45, 0.95), 4)


def madman_signal(closes: np.ndarray, volumes: np.ndarray, i: int) -> tuple:
    """Returns (direction, confidence) at bar i."""
    if i < 15:
        return "neutral", 0.5

    rsi_series = compute_rsi(closes[:i+1])
    rsi = rsi_series[-1]

    avg_vol = np.mean(volumes[max(0, i-20):i])
    vol_spike = volumes[i] > avg_vol * 2.0 if avg_vol > 0 else False

    fomo = fear = 0
    if rsi >= 70: fomo += 2
    elif rsi <= 30: fear += 2
    if vol_spike: fomo += 1

    total = fomo + fear
    if total == 0:
        return "neutral", 0.5
    if fomo > fear:
        return "buy", round(min(0.5 + (fomo / total) * 0.45, 0.95), 4)
    return "sell", round(min(0.5 + (fear / total) * 0.45, 0.95), 4)


def referee_verdict(s_dir, s_conf, m_dir, m_conf) -> str:
    if s_dir == "neutral" or s_conf < SOVEREIGN_MIN_CONF:
        return "no_signal"
    if s_dir == m_dir:
        return "crowded_skip"
    if m_dir != "neutral" and s_dir != m_dir and s_conf >= SOVEREIGN_MIN_CONF:
        return "execute"
    if m_dir == "neutral" and s_conf >= SOVEREIGN_SOLO_CONF:
        return "execute"
    return "crowded_skip"


def kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    k = (b * win_rate - q) / b
    return max(0.0, min(k * 0.5, 0.25))


# ── Single-Symbol Backtest ────────────────────────────────────────────────────

def backtest_symbol(symbol: str, df: pd.DataFrame,
                    spy_closes: np.ndarray, qqq_closes: np.ndarray,
                    equity: float) -> list:
    """
    Simulate trades on a single symbol over the full history.
    Returns list of trade dicts.
    """
    closes  = df["close"].values
    volumes = df["volume"].values
    dates   = df["timestamp"].values if "timestamp" in df.columns else df.index.values

    trades = []
    position = None  # {entry_price, entry_idx, side, qty, kelly_f}

    # Estimate kelly from first 60 bars
    if len(closes) > 60:
        ret60 = np.diff(closes[:60]) / closes[:59]
        wins60  = ret60[ret60 > 0]
        loss60  = ret60[ret60 < 0]
        wr = len(wins60) / len(ret60) if len(ret60) > 0 else 0.5
        aw = float(np.mean(wins60)) if len(wins60) > 0 else 0.01
        al = float(abs(np.mean(loss60))) if len(loss60) > 0 else 0.01
        kelly_f = kelly(wr, aw, al)
    else:
        kelly_f = 0.02

    for i in range(60, len(closes)):
        price = closes[i]

        # ── Check exit on open position ───────────────────────
        if position is not None:
            entry = position["entry_price"]
            side  = position["side"]

            pnl_pct = (price - entry) / entry if side == "buy" else (entry - price) / entry

            exit_reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            else:
                # Signal flip check — only after min hold period
                bars_held = i - position["entry_idx"]
                if bars_held >= MIN_HOLD_BARS:
                    s_dir, s_conf = sovereign_signal(closes, spy_closes, qqq_closes, i)
                    if side == "buy" and s_dir == "sell" and s_conf >= SOVEREIGN_MIN_CONF:
                        exit_reason = "signal_flip"
                    elif side == "sell" and s_dir == "buy" and s_conf >= SOVEREIGN_MIN_CONF:
                        exit_reason = "signal_flip"

            if exit_reason:
                pnl = position["qty"] * (price - entry) if side == "buy" else position["qty"] * (entry - price)
                equity += pnl
                trades.append({
                    "symbol": symbol,
                    "side": side,
                    "entry_price": round(entry, 4),
                    "exit_price": round(price, 4),
                    "qty": round(position["qty"], 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": exit_reason,
                    "entry_date": str(dates[position["entry_idx"]])[:10],
                    "exit_date": str(dates[i])[:10],
                    "bars_held": i - position["entry_idx"],
                })
                position = None
            continue  # don't enter new while in position

        # ── Check entry ───────────────────────────────────────
        s_dir, s_conf = sovereign_signal(closes, spy_closes, qqq_closes, i)
        m_dir, m_conf = madman_signal(closes, volumes, i)
        verdict = referee_verdict(s_dir, s_conf, m_dir, m_conf)

        if verdict == "execute" and s_dir != "neutral":
            if BUY_ONLY and s_dir == "sell":
                continue  # skip shorts — paper account
            is_strong = m_dir != "neutral"
            size_mult = 1.0 if is_strong else 0.6
            risk_pct  = min(BASE_RISK_PCT + kelly_f * 2, MAX_RISK_PCT) * size_mult
            qty = (equity * risk_pct) / price
            position = {
                "entry_price": price,
                "entry_idx": i,
                "side": s_dir,
                "qty": qty,
                "kelly_f": kelly_f,
            }

    return trades, equity


# ── Portfolio Backtest ────────────────────────────────────────────────────────

def run_backtest(symbols: list, days: int = 365) -> dict:
    print(f"\n{'='*60}")
    print(f"[BACKTEST] Starting — {len(symbols)} symbols, {days} days")
    print(f"[BACKTEST] Initial equity: ${INITIAL_EQUITY:,.2f}")
    print(f"{'='*60}\n")

    # Fetch benchmarks
    spy_df = fetch_history("SPY", days=days + 60)
    qqq_df = fetch_history("QQQ", days=days + 60)
    spy_closes = spy_df["close"].values if spy_df is not None else np.array([])
    qqq_closes = qqq_df["close"].values if qqq_df is not None else np.array([])

    all_trades = []
    equity = INITIAL_EQUITY
    equity_curve = [equity]

    for symbol in symbols:
        print(f"[BACKTEST] Simulating {symbol}...")
        df = fetch_history(symbol, days=days + 60)
        if df is None or len(df) < 70:
            print(f"[BACKTEST] {symbol}: insufficient data, skipping")
            continue

        # Align benchmark length to symbol
        n = len(df)
        spy_c = spy_closes[-n:] if len(spy_closes) >= n else spy_closes
        qqq_c = qqq_closes[-n:] if len(qqq_closes) >= n else qqq_closes

        trades, equity = backtest_symbol(symbol, df, spy_c, qqq_c, equity)
        all_trades.extend(trades)
        equity_curve.append(equity)
        print(f"[BACKTEST] {symbol}: {len(trades)} trades | equity now ${equity:,.2f}")

    return _compute_stats(all_trades, equity_curve, equity)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(trades: list, equity_curve: list, final_equity: float) -> dict:
    if not trades:
        print("\n[BACKTEST] No trades generated.")
        return {}

    df = pd.DataFrame(trades)
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate    = len(wins) / len(df) * 100
    total_pnl   = df["pnl"].sum()
    total_return = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    avg_win     = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss    = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else float("inf")

    # Sharpe (annualised, daily returns from equity curve)
    eq = np.array(equity_curve)
    daily_returns = np.diff(eq) / eq[:-1]
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = round((np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252), 2)

    # Max drawdown
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts().to_dict()

    print(f"\n{'='*60}")
    print(f"[BACKTEST] RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades    : {len(df)}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total P&L       : ${total_pnl:,.2f}")
    print(f"  Total return    : {total_return:+.2f}%")
    print(f"  Avg win         : ${avg_win:,.2f}")
    print(f"  Avg loss        : ${avg_loss:,.2f}")
    print(f"  Profit factor   : {profit_factor:.2f}x")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Max drawdown    : {max_dd*100:.1f}%")
    print(f"  Final equity    : ${final_equity:,.2f}")
    print(f"\n  Exit breakdown  :")
    for reason, count in exit_counts.items():
        print(f"    {reason:<20}: {count}")

    # Equity curve (ASCII sparkline)
    print(f"\n  Equity curve (${INITIAL_EQUITY/1000:.0f}k → ${final_equity/1000:.0f}k):")
    _print_sparkline(equity_curve)

    # Save trade log
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "backtest_trades.csv"
    )
    df.to_csv(log_path, index=False)
    print(f"\n  Trade log saved : {log_path}")
    print(f"{'='*60}\n")

    return {
        "total_trades": len(df),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe": sharpe,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "final_equity": round(final_equity, 2),
    }


def _print_sparkline(values: list, width: int = 50):
    if len(values) < 2:
        return
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    bars = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step]
    line = ""
    for v in sampled:
        idx = int((v - mn) / rng * (len(bars) - 1))
        line += bars[idx]
    color = "\033[92m" if values[-1] >= values[0] else "\033[91m"
    print(f"  {color}{line}\033[0m")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tug-Of-War Backtester")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--days",    type=int,  default=365)
    args = parser.parse_args()

    results = run_backtest(args.symbols, args.days)
