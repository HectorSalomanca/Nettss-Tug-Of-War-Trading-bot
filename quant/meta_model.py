"""
Meta-Model V3 — Ensemble Scorer with Feature Neutralization

Combines 3 signal streams into a final confidence score:
  final_score = w1 * stockformer_score + w2 * ofi_regression + w3 * hmm_posterior

Weights learned via Ridge regression on last 30 days of trade outcomes.
50% feature neutralization against SPY beta applied to final_score.

Used by the Referee to replace raw sovereign_confidence in verdict logic.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from referee.secret_vault import get_secret
from supabase import create_client, Client

SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_secret("SUPABASE_SERVICE_KEY") or get_secret("SUPABASE_ANON_KEY")
USER_ID           = get_secret("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Default weights (used before Ridge regression has enough data)
DEFAULT_WEIGHTS = {
    "stockformer": 0.35,   # Transformer conviction (feature-neutralized)
    "ofi":         0.25,   # OFI regression score (cross-sectional deviated)
    "hmm":         0.20,   # HMM regime posterior
    "alpha":       0.20,   # WorldQuant 101 formulaic alpha composite
}

NEUTRALIZATION_PCT = 0.0   # Disabled: neutralization was halving all scores (same bug as sovereign)


# ── OFI Regression ───────────────────────────────────────────────────────────

def ofi_to_score(ofi_z: float, iceberg: bool, stacked: bool, trapped: bool) -> float:
    """
    Convert OFI microstructure signals into a single regression-like score [-1, 1].
    Positive = bullish pressure, Negative = bearish pressure.
    """
    score = 0.0

    # Base OFI Z-score contribution (clamped)
    score += np.clip(ofi_z * 0.2, -0.5, 0.5)

    # Iceberg = institutional absorption = bearish for retail
    if iceberg:
        score -= 0.3

    # Stacked imbalance = strong directional retail momentum
    if stacked:
        score += 0.25

    # Trapped exhaustion = retail trapped at peak = bearish
    if trapped:
        score -= 0.35

    return float(np.clip(score, -1.0, 1.0))


# ── HMM Posterior Score ──────────────────────────────────────────────────────

def hmm_to_score(regime: str, confidence: float, state_v3: str = "") -> float:
    """
    Convert HMM regime state + confidence into a directional score [-1, 1].
    """
    if regime == "crisis":
        return -1.0 * confidence
    elif state_v3 == "trend_bear" or (regime == "trend" and state_v3 == "trend_bear"):
        return -0.5 * confidence
    elif state_v3 == "trend_bull" or regime == "trend":
        return 0.5 * confidence
    elif regime == "chop":
        return 0.0  # neutral in chop
    return 0.0


# ── Feature Neutralization ───────────────────────────────────────────────────

def neutralize_score(raw_score: float, spy_beta_exposure: float = 0.0) -> float:
    """
    Apply fractional feature neutralization.
    Removes NEUTRALIZATION_PCT of the score's correlation with SPY beta.

    neutralized = raw - neutralization_pct * spy_beta_exposure
    """
    neutralized = raw_score - NEUTRALIZATION_PCT * spy_beta_exposure
    return float(np.clip(neutralized, -1.0, 1.0))


# ── Ridge Weight Learning ────────────────────────────────────────────────────

_learned_weights = None

def learn_weights_from_history() -> dict:
    """
    Fit Ridge regression on labeled trades to learn optimal ensemble weights.
    Requires tbl_label column in trades table (1=win, -1=loss, 0=time_stop).
    Falls back to DEFAULT_WEIGHTS if fewer than 20 labeled samples.
    """
    global _learned_weights

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[META] scikit-learn not installed — using default weights")
        return DEFAULT_WEIGHTS

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        # Fetch labeled filled trades with their tug_result_id for component lookup
        result = (
            supabase.table("trades")
            .select("tug_result_id, symbol, side, tbl_label, created_at")
            .gte("created_at", cutoff)
            .eq("status", "filled")
            .not_.is_("tbl_label", "null")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        trades = result.data or []
        if len(trades) < 20:
            print(f"[META] Only {len(trades)} labeled trades — need 20+ for Ridge, using defaults")
            return DEFAULT_WEIGHTS

        # Fetch tug_results for component scores
        tug_ids = [t["tug_result_id"] for t in trades if t.get("tug_result_id")]
        if not tug_ids:
            print("[META] No tug_result_ids found — using defaults")
            return DEFAULT_WEIGHTS

        tug_result = (
            supabase.table("tug_results")
            .select("id, stockformer_score, ofi_score, hmm_score")
            .in_("id", tug_ids[:100])
            .execute()
        )
        tug_map = {r["id"]: r for r in (tug_result.data or [])}

        X_rows, y_rows = [], []
        for trade in trades:
            tid = trade.get("tug_result_id")
            tug = tug_map.get(tid)
            if not tug:
                continue
            sf  = float(tug.get("stockformer_score") or 0.0)
            ofi = float(tug.get("ofi_score") or 0.0)
            hmm = float(tug.get("hmm_score") or 0.0)
            label = int(trade["tbl_label"])
            # Convert label: 1=win → +1, -1=loss → -1, 0=time_stop → 0
            X_rows.append([sf, ofi, hmm])
            y_rows.append(float(label))

        if len(X_rows) < 20:
            print(f"[META] Only {len(X_rows)} matched rows — using defaults")
            return DEFAULT_WEIGHTS

        X = np.array(X_rows)
        y = np.array(y_rows)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_scaled, y)

        raw_coefs = ridge.coef_  # [sf_weight, ofi_weight, hmm_weight]

        # Softmax-normalise to sum=1, all positive
        exp_coefs = np.exp(np.clip(raw_coefs, -5, 5))
        weights_arr = exp_coefs / exp_coefs.sum()

        learned = {
            "stockformer": round(float(weights_arr[0]), 4),
            "ofi":         round(float(weights_arr[1]), 4),
            "hmm":         round(float(weights_arr[2]), 4),
        }
        print(f"[META] Ridge fitted on {len(X_rows)} samples → weights={learned}")
        return learned

    except Exception as e:
        print(f"[META] Ridge fitting error: {e} — using defaults")
        return DEFAULT_WEIGHTS


def get_weights() -> dict:
    """Return current ensemble weights (learned or default)."""
    global _learned_weights
    if _learned_weights is None:
        _learned_weights = learn_weights_from_history()
    return _learned_weights


# ── Ensemble Scoring ─────────────────────────────────────────────────────────

def compute_ensemble_score(
    stockformer_score: float,
    ofi_z: float,
    iceberg: bool,
    stacked: bool,
    trapped: bool,
    regime: str,
    regime_confidence: float,
    state_v3: str = "",
    spy_beta: float = 0.0,
    alpha_composite: float = 0.0,
) -> dict:
    """
    Compute the final ensemble meta-model score.

    Returns dict with:
      - ensemble_score: [-1, 1] final conviction
      - ensemble_direction: "buy" / "sell" / "neutral"
      - ensemble_confidence: [0.5, 0.95] mapped confidence
      - component scores for logging
    """
    weights = get_weights()

    # Component scores
    sf_score    = float(np.clip(stockformer_score, -1.0, 1.0))
    ofi_score   = ofi_to_score(ofi_z, iceberg, stacked, trapped)
    hmm_score   = hmm_to_score(regime, regime_confidence, state_v3)
    alpha_score = float(np.clip(alpha_composite, -1.0, 1.0))

    # Weighted ensemble (4 streams)
    raw_score = (
        weights["stockformer"] * sf_score +
        weights["ofi"]         * ofi_score +
        weights["hmm"]         * hmm_score +
        weights.get("alpha", 0.20) * alpha_score
    )

    final_score = float(np.clip(raw_score, -1.0, 1.0))

    # Map to direction + confidence
    if final_score > 0.05:
        direction = "buy"
        confidence = round(min(0.5 + abs(final_score) * 0.45, 0.95), 4)
    elif final_score < -0.05:
        direction = "sell"
        confidence = round(min(0.5 + abs(final_score) * 0.45, 0.95), 4)
    else:
        direction = "neutral"
        confidence = 0.5

    return {
        "ensemble_score": round(final_score, 4),
        "ensemble_direction": direction,
        "ensemble_confidence": confidence,
        "stockformer_component": round(sf_score, 4),
        "ofi_component": round(ofi_score, 4),
        "hmm_component": round(hmm_score, 4),
        "alpha_component": round(alpha_score, 4),
        "weights": weights,
        "spy_beta": round(spy_beta, 4),
        "neutralization_pct": NEUTRALIZATION_PCT,
    }


if __name__ == "__main__":
    # Test with sample inputs
    result = compute_ensemble_score(
        stockformer_score=0.3,
        ofi_z=1.5,
        iceberg=False,
        stacked=True,
        trapped=False,
        regime="trend",
        regime_confidence=0.85,
        state_v3="trend_bull",
        spy_beta=0.1,
    )
    print(f"Ensemble: {result['ensemble_direction'].upper()} @ {result['ensemble_confidence']:.2%}")
    print(f"  Score: {result['ensemble_score']:.4f}")
    print(f"  SF={result['stockformer_component']:.4f} | OFI={result['ofi_component']:.4f} | HMM={result['hmm_component']:.4f}")
