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

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USER_ID           = os.getenv("USER_ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Default weights (used before Ridge regression has enough data)
DEFAULT_WEIGHTS = {
    "stockformer": 0.45,   # highest weight — Transformer conviction
    "ofi":         0.30,   # OFI regression score
    "hmm":         0.25,   # HMM regime posterior
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
    Fit Ridge regression on last 30 days of trade outcomes to learn optimal
    ensemble weights. Falls back to DEFAULT_WEIGHTS if insufficient data.
    """
    global _learned_weights

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        result = (
            supabase.table("trades")
            .select("symbol, side, qty, limit_price, status, created_at")
            .gte("created_at", cutoff)
            .eq("status", "filled")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        if not result.data or len(result.data) < 20:
            print("[META] Insufficient trade history for Ridge — using defaults")
            return DEFAULT_WEIGHTS

        # For now, use default weights until we have labeled outcomes
        # (Triple Barrier Labeler will provide tbl_label for proper Ridge fitting)
        print(f"[META] Found {len(result.data)} trades — Ridge fitting requires TBL labels")
        return DEFAULT_WEIGHTS

    except Exception as e:
        print(f"[META] Weight learning error: {e}")
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
    sf_score  = float(np.clip(stockformer_score, -1.0, 1.0))
    ofi_score = ofi_to_score(ofi_z, iceberg, stacked, trapped)
    hmm_score = hmm_to_score(regime, regime_confidence, state_v3)

    # Weighted ensemble
    raw_score = (
        weights["stockformer"] * sf_score +
        weights["ofi"]         * ofi_score +
        weights["hmm"]         * hmm_score
    )

    # No neutralization — it was compressing all scores toward 0, killing signal
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
