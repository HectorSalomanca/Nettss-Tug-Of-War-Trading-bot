"""
Stockformer V3 — Multi-Asset Transformer for Cross-Sectional Momentum

Architecture:
  - Input: [batch, seq_len=60, n_assets=10, features=4]
    Features per asset per day: AFD-diff close, volume, pure alpha, realized vol
  - STL decomposition: splits trend + seasonal before attention
  - Temporal Self-Attention: compares every timestamp against every other
  - Cross-Asset Attention: captures lead-lag relationships (NVDA→CRWD, TSMC→MSTR)
  - Loss: negative Sharpe ratio (trains directly on risk-adjusted returns)
  - Output: per-asset conviction score [-1, 1]

Runs on MPS (M1 GPU) when available, falls back to CPU.
Retrained weekly; weights saved to models/stockformer.pt
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee.secret_vault import get_secret

# ── Device Selection ─────────────────────────────────────────────────────────

def get_device():
    if not HAS_TORCH:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stockformer.pt")

# ── Config ───────────────────────────────────────────────────────────────────

WATCHLIST = [
    "NVDA", "CRWD", "LLY", "TSMC", "JPM", "NEE", "CAT", "SONY", "PLTR", "MSTR",
    "MSFT", "AMZN", "META", "GLD", "XLE", "UBER", "AMD", "COIN", "MRNA", "IWM",
]
SEQ_LEN      = 60     # 60-day lookback
N_ASSETS     = 20
N_FEATURES   = 4      # afd_close, volume_norm, pure_alpha, realized_vol
D_MODEL      = 64     # transformer hidden dim
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.1
LR           = 1e-3
EPOCHS       = 50
BATCH_SIZE   = 16

# Lead-lag attention bias (hard-coded from cross-sectional research)
# Index mapping:
#   NVDA=0, CRWD=1, LLY=2, TSMC=3, JPM=4, NEE=5, CAT=6, SONY=7, PLTR=8, MSTR=9
#   MSFT=10, AMZN=11, META=12, GLD=13, XLE=14, UBER=15, AMD=16, COIN=17, MRNA=18, IWM=19
LEAD_LAG_PAIRS = {
    (0, 1):  0.30,  # NVDA → CRWD (tech stack flow)
    (3, 9):  0.20,  # TSMC → MSTR (macro liquidity)
    (2, 2):  0.10,  # LLY self-reinforcing (pharma momentum)
    (0, 3):  0.15,  # NVDA → TSMC (supply chain)
    (8, 9):  0.10,  # PLTR → MSTR (high-beta correlation)
    (0, 16): 0.35,  # NVDA → AMD (semiconductor sector)
    (16, 0): 0.20,  # AMD → NVDA (bidirectional semi)
    (10, 11):0.15,  # MSFT → AMZN (mega-cap tech rotation)
    (9, 17): 0.30,  # MSTR → COIN (crypto proxy)
    (17, 9): 0.25,  # COIN → MSTR (crypto proxy reverse)
    (13, 14):0.20,  # GLD → XLE (macro risk-off rotation)
    (19, 4): 0.15,  # IWM → JPM (small-cap risk-on → financials)
    (12, 15):0.15,  # META → UBER (consumer sentiment)
    (2, 18): 0.20,  # LLY → MRNA (biotech sector flow)
}


# ── Stockformer Model ────────────────────────────────────────────────────────

if HAS_TORCH:

    class TemporalAttention(nn.Module):
        """Self-attention across the time dimension for each asset."""
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: [batch, seq_len, d_model]
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + self.dropout(attn_out))


    class CrossAssetAttention(nn.Module):
        """Self-attention across the asset dimension at each timestep."""
        def __init__(self, d_model, n_heads, n_assets, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # Learnable lead-lag bias matrix
            self.lead_lag_bias = nn.Parameter(torch.zeros(n_assets, n_assets))
            # Initialize with known lead-lag pairs
            with torch.no_grad():
                for (leader, lagger), weight in LEAD_LAG_PAIRS.items():
                    self.lead_lag_bias[leader, lagger] = weight

        def forward(self, x):
            # x: [batch, n_assets, d_model]
            attn_out, _ = self.attn(x, x, x, attn_mask=None)
            # Add lead-lag bias
            bias = self.lead_lag_bias.unsqueeze(0).expand(x.size(0), -1, -1)
            biased = attn_out + torch.bmm(bias, x)
            return self.norm(x + self.dropout(biased))


    class StockformerBlock(nn.Module):
        """One Stockformer layer: temporal attention → cross-asset attention → FFN."""
        def __init__(self, d_model, n_heads, n_assets, dropout=0.1):
            super().__init__()
            self.temporal = TemporalAttention(d_model, n_heads, dropout)
            self.cross_asset = CrossAssetAttention(d_model, n_heads, n_assets, dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # x: [batch, seq_len, n_assets, d_model]
            B, T, A, D = x.shape

            # Temporal attention per asset
            x_temp = x.permute(0, 2, 1, 3).reshape(B * A, T, D)
            x_temp = self.temporal(x_temp)
            x_temp = x_temp.reshape(B, A, T, D).permute(0, 2, 1, 3)

            # Cross-asset attention per timestep
            x_cross = x_temp.reshape(B * T, A, D)
            x_cross = self.cross_asset(x_cross)
            x_cross = x_cross.reshape(B, T, A, D)

            # FFN
            out = self.norm(x_cross + self.ffn(x_cross))
            return out


    class Stockformer(nn.Module):
        """
        Full Stockformer: projects input features → transformer blocks → conviction scores.
        Output: [batch, n_assets] with values in [-1, 1].
        """
        def __init__(self, n_features=N_FEATURES, d_model=D_MODEL, n_heads=N_HEADS,
                     n_layers=N_LAYERS, n_assets=N_ASSETS, seq_len=SEQ_LEN, dropout=DROPOUT):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, 1, d_model) * 0.02)
            self.asset_embed = nn.Parameter(torch.randn(1, 1, n_assets, d_model) * 0.02)

            self.blocks = nn.ModuleList([
                StockformerBlock(d_model, n_heads, n_assets, dropout)
                for _ in range(n_layers)
            ])

            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh(),  # output in [-1, 1]
            )

        def forward(self, x):
            # x: [batch, seq_len, n_assets, n_features]
            B, T, A, F = x.shape
            x = self.input_proj(x)  # [B, T, A, D]
            x = x + self.pos_embed[:, :T, :, :] + self.asset_embed[:, :, :A, :]

            for block in self.blocks:
                x = block(x)

            # Pool over time (use last timestep)
            x_last = x[:, -1, :, :]  # [B, A, D]
            scores = self.head(x_last).squeeze(-1)  # [B, A]
            return scores


    def sharpe_loss(predictions, returns):
        """
        Negative Sharpe ratio loss.
        predictions: [batch, n_assets] — conviction scores
        returns: [batch, n_assets] — actual next-day returns
        """
        portfolio_returns = (predictions * returns).sum(dim=1)
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + 1e-8
        return -(mean_ret / std_ret)


# ── Data Preparation ─────────────────────────────────────────────────────────

def prepare_training_data(
    all_data: dict,  # {symbol: pd.DataFrame with columns [close, volume, afd_close, pure_alpha, realized_vol]}
) -> Tuple:
    """
    Build training tensors from historical data.
    Returns (X, Y) where:
      X: [n_samples, SEQ_LEN, N_ASSETS, N_FEATURES]
      Y: [n_samples, N_ASSETS] — next-day returns
    """
    if not HAS_TORCH:
        return None, None

    # Align all symbols to same dates
    aligned = {}
    common_dates = None
    for sym in WATCHLIST:
        if sym not in all_data or all_data[sym] is None:
            return None, None
        df = all_data[sym].copy()
        dates = set(df.index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates & dates
        aligned[sym] = df

    if common_dates is None or len(common_dates) < SEQ_LEN + 10:
        return None, None

    common_dates = sorted(common_dates)

    # Build feature tensor
    n_days = len(common_dates)
    features = np.zeros((n_days, N_ASSETS, N_FEATURES))
    returns = np.zeros((n_days, N_ASSETS))

    for j, sym in enumerate(WATCHLIST):
        df = aligned[sym].loc[common_dates]
        closes = df["close"].values

        # Feature 0: AFD-differenced close (normalized)
        afd = df["afd_close"].values if "afd_close" in df.columns else np.diff(closes, prepend=closes[0]) / (closes + 1e-8)
        features[:, j, 0] = (afd - np.nanmean(afd)) / (np.nanstd(afd) + 1e-8)

        # Feature 1: Volume (log-normalized)
        vol = np.log1p(df["volume"].values)
        features[:, j, 1] = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)

        # Feature 2: Pure alpha
        pa = df["pure_alpha"].values if "pure_alpha" in df.columns else np.zeros(n_days)
        features[:, j, 2] = (pa - np.nanmean(pa)) / (np.nanstd(pa) + 1e-8)

        # Feature 3: Realized vol (20-day rolling)
        rets = np.diff(closes, prepend=closes[0]) / (closes + 1e-8)
        rv = pd.Series(rets).rolling(20).std().fillna(0).values
        features[:, j, 3] = (rv - np.mean(rv)) / (np.std(rv) + 1e-8)

        # Target: next-day return
        next_ret = np.zeros(n_days)
        next_ret[:-1] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-8)
        returns[:, j] = next_ret

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Build sliding windows
    X_list, Y_list = [], []
    for i in range(SEQ_LEN, n_days - 1):
        X_list.append(features[i - SEQ_LEN:i])  # [SEQ_LEN, N_ASSETS, N_FEATURES]
        Y_list.append(returns[i])                # [N_ASSETS]

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return X, Y


# ── Training ─────────────────────────────────────────────────────────────────

def train_stockformer(all_data: dict, epochs: int = EPOCHS) -> Optional[object]:
    """Train the Stockformer on historical data. Returns trained model."""
    if not HAS_TORCH:
        print("[STOCKFORMER] PyTorch not installed — skipping training")
        return None

    X, Y = prepare_training_data(all_data)
    if X is None or len(X) < BATCH_SIZE * 2:
        print("[STOCKFORMER] Insufficient training data")
        return None

    print(f"[STOCKFORMER] Training on {len(X)} samples, device={DEVICE}")

    model = Stockformer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train/val split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            xb = X_train[batch_idx].to(DEVICE)
            yb = Y_train[batch_idx].to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = sharpe_loss(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(DEVICE))
            val_loss = sharpe_loss(val_preds, Y_val.to(DEVICE)).item()

        avg_train = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"[STOCKFORMER] Epoch {epoch+1}/{epochs} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[STOCKFORMER] Early stopping at epoch {epoch+1}")
                break

    print(f"[STOCKFORMER] Training complete. Best val_loss={best_val_loss:.4f}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model


# ── Inference ────────────────────────────────────────────────────────────────

_cached_model = None

def load_model() -> Optional[object]:
    """Load trained model from disk. Caches in memory."""
    global _cached_model
    if not HAS_TORCH:
        return None
    if _cached_model is not None:
        return _cached_model
    if not os.path.exists(MODEL_PATH):
        print("[STOCKFORMER] No trained model found — run training first")
        return None
    try:
        model = Stockformer().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _cached_model = model
        print(f"[STOCKFORMER] Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[STOCKFORMER] Model load error: {e}")
        return None


def _neutralize_predictions(raw_scores: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Numerai-style Feature Neutralization.

    Regress raw model predictions against input features, then subtract
    the linear component. This forces the Stockformer to only output
    non-linear alpha — the hidden relationships it found that simple
    momentum/vol/OFI can't explain.

    raw_scores: [N_ASSETS] — raw conviction scores from Stockformer
    features:   [N_ASSETS, N_FEATURES] — last-timestep input features

    Returns: [N_ASSETS] — orthogonalized scores
    """
    if features.shape[0] < 3 or np.std(raw_scores) < 1e-8:
        return raw_scores

    try:
        # OLS: raw_scores = beta @ features.T + residual
        X = np.column_stack([np.ones(features.shape[0]), features])
        beta, _, _, _ = np.linalg.lstsq(X, raw_scores, rcond=None)
        linear_component = X @ beta
        residual = raw_scores - linear_component

        # Scale residual back to similar magnitude as raw scores
        if np.std(residual) > 1e-8:
            residual = residual * (np.std(raw_scores) / np.std(residual))

        return np.clip(residual, -1.0, 1.0)
    except Exception:
        return raw_scores


def predict(features_dict: dict) -> dict:
    """
    Run inference on current market data with Numerai-style feature neutralization.

    Args:
        features_dict: {symbol: np.ndarray of shape [SEQ_LEN, N_FEATURES]}

    Returns:
        {symbol: conviction_score} where score is in [-1, 1]
        Scores are orthogonalized against input features to remove
        linear factor exposure (momentum, vol, etc).
    """
    if not HAS_TORCH:
        return {sym: 0.0 for sym in WATCHLIST}

    model = load_model()
    if model is None:
        return {sym: 0.0 for sym in WATCHLIST}

    # Build input tensor [1, SEQ_LEN, N_ASSETS, N_FEATURES]
    x = np.zeros((1, SEQ_LEN, N_ASSETS, N_FEATURES))
    for j, sym in enumerate(WATCHLIST):
        if sym in features_dict and features_dict[sym] is not None:
            data = features_dict[sym]
            if len(data) >= SEQ_LEN:
                x[0, :, j, :] = data[-SEQ_LEN:]
            elif len(data) > 0:
                x[0, -len(data):, j, :] = data

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        raw_scores = model(x_tensor).cpu().numpy()[0]  # [N_ASSETS]

    # Numerai Feature Neutralization: orthogonalize against last-timestep features
    last_features = x[0, -1, :, :]  # [N_ASSETS, N_FEATURES]
    neutralized = _neutralize_predictions(raw_scores, last_features)

    result = {}
    for j, sym in enumerate(WATCHLIST):
        result[sym] = float(np.clip(neutralized[j], -1.0, 1.0))

    return result


# ── Auto-Retrain Entry Point ─────────────────────────────────────────────────

def retrain_stockformer():
    """
    Fetch 180 days of daily bars for all watchlist symbols,
    build training data, and retrain the Stockformer model.
    Called automatically every Sunday by the engine scheduler.
    """
    if not HAS_TORCH:
        print("[STOCKFORMER] PyTorch not available — skipping retrain")
        return

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    data_client = StockHistoricalDataClient(
        get_secret("ALPACA_API_KEY"), get_secret("ALPACA_SECRET_KEY")
    )

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=180)

    all_data = {}
    for sym in WATCHLIST:
        alpaca_sym = "TSM" if sym == "TSMC" else sym
        try:
            req = StockBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=TimeFrame.Day,
                start=start, end=end,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(alpaca_sym, level=0)
            df = df.sort_index().copy()

            # Compute AFD-close and pure_alpha approximations
            closes = df["close"].values
            df["afd_close"] = np.diff(closes, prepend=closes[0]) / (closes + 1e-8)
            df["pure_alpha"] = df["afd_close"] - df["afd_close"].rolling(20).mean().fillna(0)
            all_data[sym] = df
            print(f"[STOCKFORMER RETRAIN] {sym}: {len(df)} days fetched")
        except Exception as e:
            print(f"[STOCKFORMER RETRAIN] {sym}: fetch error — {e}")

    if len(all_data) < N_ASSETS:
        print(f"[STOCKFORMER RETRAIN] Only {len(all_data)}/{N_ASSETS} symbols fetched — aborting")
        return

    print(f"[STOCKFORMER RETRAIN] Starting training on {N_ASSETS} assets...")
    model = train_stockformer(all_data)
    if model is not None:
        # Invalidate in-memory cache so next predict() loads fresh weights
        global _cached_model
        _cached_model = None
        print("[STOCKFORMER RETRAIN] Complete — new weights saved, cache invalidated")
    else:
        print("[STOCKFORMER RETRAIN] Training failed")


if __name__ == "__main__":
    print(f"[STOCKFORMER] Device: {DEVICE}")
    print(f"[STOCKFORMER] PyTorch available: {HAS_TORCH}")
    if HAS_TORCH:
        model = Stockformer()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[STOCKFORMER] Model params: {total_params:,}")
        dummy = torch.randn(2, SEQ_LEN, N_ASSETS, N_FEATURES)
        out = model(dummy)
        print(f"[STOCKFORMER] Output shape: {out.shape} (expected [2, {N_ASSETS}])")
        print(f"[STOCKFORMER] Sample scores: {out[0].detach().numpy()}")
