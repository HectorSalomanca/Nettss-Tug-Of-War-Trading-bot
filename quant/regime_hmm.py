"""
Regime HMM — 3-State Gaussian Hidden Markov Model

States:
  0 = Chop   (low volatility, mean-reverting, range-bound)
  1 = Trend  (directional momentum, normal vol)
  2 = Crisis (volatility spike, fat tails, all trading halted)

Trained on SPY daily returns using volatility + momentum features.
Updates every 15 minutes. Writes current state to Supabase.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

STATE_NAMES = {0: "chop", 1: "trend", 2: "crisis"}
STATE_LABELS = {"chop": 0, "trend": 1, "crisis": 2}


def get_spy_history(days: int = 252) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs("SPY", level=0)
        return df.sort_index()
    except Exception as e:
        print(f"[HMM] SPY fetch error: {e}")
        return None


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Two features per day:
      - Realized volatility (20-day rolling std of returns, annualized)
      - 5-day momentum (return over last 5 days)
    """
    closes = df["close"].values
    returns = np.diff(closes) / closes[:-1]

    vol_window = 20
    mom_window = 5
    n = len(returns)

    features = []
    for i in range(vol_window, n):
        vol = np.std(returns[i - vol_window:i]) * np.sqrt(252)
        mom = (closes[i + 1] - closes[i + 1 - mom_window]) / closes[i + 1 - mom_window]
        features.append([vol, mom])

    return np.array(features)


def fit_hmm(features: np.ndarray) -> object:
    """Fit a 3-state Gaussian HMM. Returns fitted model."""
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X)
        return model, scaler
    except Exception as e:
        print(f"[HMM] Fit error: {e}")
        return None, None


def _assign_state_labels(model, scaler) -> dict:
    """
    Map HMM hidden states 0/1/2 to chop/trend/crisis
    by inspecting the mean volatility of each state's emissions.
    Highest vol = crisis, lowest vol = chop, middle = trend.
    """
    try:
        means = scaler.inverse_transform(model.means_)
        vol_by_state = {i: means[i][0] for i in range(3)}
        sorted_states = sorted(vol_by_state, key=vol_by_state.get)
        return {
            sorted_states[0]: "chop",
            sorted_states[1]: "trend",
            sorted_states[2]: "crisis",
        }
    except Exception:
        return {0: "chop", 1: "trend", 2: "crisis"}


def infer_current_regime() -> dict:
    """
    Full pipeline: fetch SPY → build features → fit HMM → decode current state.
    Returns dict with state name, confidence, and raw features.
    """
    df = get_spy_history(days=252)
    if df is None or len(df) < 60:
        print("[HMM] Insufficient data — defaulting to trend")
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    features = build_feature_matrix(df)
    if len(features) < 30:
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    model, scaler = fit_hmm(features)
    if model is None:
        return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.15, "spy_momentum": 0.0}

    state_map = _assign_state_labels(model, scaler)

    X_scaled = scaler.transform(features)
    hidden_states = model.predict(X_scaled)
    posteriors = model.predict_proba(X_scaled)

    current_hidden = int(hidden_states[-1])
    current_state = state_map[current_hidden]
    confidence = float(posteriors[-1][current_hidden])

    current_vol  = float(features[-1][0])
    current_mom  = float(features[-1][1])

    print(f"[HMM] Regime: {current_state.upper()} (conf={confidence:.2%}, vol={current_vol:.3f}, mom={current_mom:.4f})")

    return {
        "state": current_state,
        "confidence": round(confidence, 4),
        "spy_volatility": round(current_vol, 4),
        "spy_momentum": round(current_mom, 6),
    }


def log_regime(regime: dict):
    """Write current regime state to Supabase."""
    if not USER_ID:
        return
    try:
        supabase.table("regime_states").insert({
            "state": regime["state"],
            "confidence": regime["confidence"],
            "spy_volatility": regime["spy_volatility"],
            "spy_momentum": regime["spy_momentum"],
            "user_id": USER_ID,
        }).execute()
    except Exception as e:
        print(f"[HMM] Supabase log error: {e}")


def get_latest_regime() -> str:
    """
    Fast path: read the most recent regime from Supabase.
    Falls back to 'trend' if no record found.
    """
    r = get_latest_regime_full()
    return r["state"]


def get_latest_regime_full() -> dict:
    """
    Returns full regime dict: {state, confidence, spy_volatility, spy_momentum}.
    Falls back to trend/0.5 defaults if no record found.
    """
    try:
        result = (
            supabase.table("regime_states")
            .select("state, confidence, spy_volatility, spy_momentum, created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            age_seconds = (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            if age_seconds < 1800:
                return {
                    "state":          row["state"],
                    "confidence":     row.get("confidence", 0.5),
                    "spy_volatility": row.get("spy_volatility", 0.0),
                    "spy_momentum":   row.get("spy_momentum", 0.0),
                }
    except Exception:
        pass
    return {"state": "trend", "confidence": 0.5, "spy_volatility": 0.0, "spy_momentum": 0.0}


def run_and_log() -> dict:
    """Infer regime and log it. Called by the scheduler."""
    regime = infer_current_regime()
    log_regime(regime)
    return regime


if __name__ == "__main__":
    result = run_and_log()
    print(result)
