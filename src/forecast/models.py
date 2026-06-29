"""Track B models: Holt-Winters baseline + LSTM + Ensemble (D9).

Both models consume sliding windows of per-AR time-series:
  • HoltWintersForecaster — Holt's double exponential smoothing on the
    historical LABEL sequence within the lookback window.  No training step;
    parameters (alpha, beta) come from config.  Works on any AR at inference.
  • LSTMForecaster — 2-layer LSTM on the FEATURE matrix.  Trained with
    BCEWithLogitsLoss + pos_weight (TRAIN only).
  • Ensemble — weighted average: P = α·P_hw + (1-α)·P_lstm.

Device and path conventions come from src.config (never hard-coded here).
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Holt-Winters (double exponential smoothing — level + trend, no seasonal)
# ---------------------------------------------------------------------------

class HoltWintersForecaster:
    """Per-window Holt's double exponential smoothing on historical labels.

    At prediction time for a window of length L, fits the smoothing recurrence
    online (O(L) per sample) with fixed alpha (level) and beta (trend) from
    config — no per-window optimisation, which keeps inference fast.

    Parameters
    ----------
    alpha : float  Level smoothing factor ∈ (0, 1).
    beta  : float  Trend smoothing factor ∈ (0, 1).
    clip  : bool   Clip the forecast to [0, 1]. Use True when smoothing a binary
                   label history (forecast == probability); use False when
                   smoothing a real-valued scalar FEATURE series (D9) — the raw
                   forecast is then mapped to a probability by a calibrator.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, clip: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.clip = clip

    def _holt_next(self, series: np.ndarray) -> float:
        """Forecast the next step from a 1-D series (Holt's double smoothing)."""
        h = series.astype(float)
        if h.size == 0:
            return 0.0
        if h.size == 1:
            v = float(h[0])
            return float(np.clip(v, 0.0, 1.0)) if self.clip else v
        L = h[0]
        B = h[1] - h[0]
        for i in range(1, h.size):
            L_new = self.alpha * h[i] + (1.0 - self.alpha) * (L + B)
            B = self.beta * (L_new - L) + (1.0 - self.beta) * B
            L = L_new
        forecast = L + B
        return float(np.clip(forecast, 0.0, 1.0)) if self.clip else float(forecast)

    def predict_batch(self, series_windows: np.ndarray) -> np.ndarray:
        """Vectorised one-step forecast over a batch of scalar series.

        Parameters
        ----------
        series_windows : (N, T) array — one scalar series per sample (a binary
            label history, or a real-valued feature series for D9).

        Returns
        -------
        (N,) float array of forecasts (probabilities if clip=True, else raw).
        """
        return np.array([self._holt_next(w) for w in series_windows], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class _WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))   # (N, L, F)
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# LSTM forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """2-layer LSTM on SHARP parameter windows → binary flare probability.

    Input:  (batch, lookback, n_features)
    Output: (batch, 1)  logit  (apply sigmoid to get probability)

    Use .fit() to train, .predict_proba() for inference.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → logit

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cfg: dict,
        device: str = "cpu",
    ) -> None:
        """Train the LSTM on oversampled training windows.

        Parameters
        ----------
        X_train : (N, lookback, n_features) float32 — already normalised.
        y_train : (N,) int/float binary labels.
        cfg     : Config dict (keys: lstm_lr, lstm_batch_size, lstm_epochs).
        device  : 'cpu' or 'cuda'.
        """
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        pos_weight = torch.tensor(
            [neg / max(pos, 1)], dtype=torch.float32
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(cfg.get("lstm_lr", 1e-3))
        )

        dataset = _WindowDataset(X_train, y_train)
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.get("lstm_batch_size", 256)),
            shuffle=True,
            drop_last=False,
        )

        n_epochs = int(cfg.get("lstm_epochs", 30))
        self.to(device)
        self.train()
        for epoch in range(n_epochs):
            total_loss, n_seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logit = self(xb)
                loss = criterion(logit, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                n_seen += len(xb)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "LSTM epoch %3d/%d  loss=%.4f",
                    epoch + 1, n_epochs, total_loss / max(n_seen, 1),
                )
        self.eval()
        logger.info("LSTM training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return flare probabilities ∈ [0, 1] for a batch of windows.

        Parameters
        ----------
        X : (N, lookback, n_features) float32.

        Returns
        -------
        (N,) float32 probability array.
        """
        self.eval()
        dummy_y = np.zeros(len(X), dtype=np.float32)
        loader = DataLoader(
            _WindowDataset(X, dummy_y),
            batch_size=batch_size,
            shuffle=False,
        )
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                logit = self(xb.to(device))
                probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(probs).ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class Ensemble:
    """Weighted-average ensemble of HW and LSTM probability outputs.

    P_ensemble = alpha * P_hw + (1 - alpha) * P_lstm

    Parameters
    ----------
    hw            : HoltWintersForecaster
    lstm          : LSTMForecaster
    alpha         : float — weight for the HW component (default 0.5).
    hw_calibrator : optional callable mapping HW raw forecasts (N,) → probs (N,).
                    Required when HW smooths a real-valued feature series (D9);
                    omit when HW already outputs probabilities (clip=True).
    """

    def __init__(
        self,
        hw: HoltWintersForecaster,
        lstm: LSTMForecaster,
        alpha: float = 0.5,
        hw_calibrator=None,
    ):
        self.hw = hw
        self.lstm = lstm
        self.alpha = float(alpha)
        self.hw_calibrator = hw_calibrator

    def predict_proba(
        self,
        hw_series: np.ndarray,
        feat_windows: np.ndarray,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> np.ndarray:
        """Ensemble prediction.

        Parameters
        ----------
        hw_series    : (N, T) — scalar series per sample for the HW component.
        feat_windows : (N, T, n_features) — SHARP features for the LSTM.

        Returns
        -------
        (N,) float32 ensemble probability array.
        """
        p_hw = self.hw.predict_batch(hw_series)
        if self.hw_calibrator is not None:
            p_hw = self.hw_calibrator(p_hw)
        p_lstm = self.lstm.predict_proba(feat_windows, device=device, batch_size=batch_size)
        return (self.alpha * p_hw + (1.0 - self.alpha) * p_lstm).astype(np.float32)
