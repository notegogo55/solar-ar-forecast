"""Track B — Holt-Winters + LSTM ensemble forecaster (D9).

Develop & VALIDATE on SWAN-SF first (no images needed), then swap in real
extracted features. Runs LOCAL on the 1650/CPU — does not touch Kaggle.
Enforces the strict validation protocol (D8) via src.eval.

SWAN-SF data model
------------------
Each SWAN-SF *instance* file is ONE ready-made fixed-length labelled sample
(a multivariate window → "≥M1.0 flare in the prediction window?", D7). The 5
partitions are non-overlapping TIME blocks, so the canonical D8 split is at the
partition level (train p1–4, test p5) with a whole-AR guard. No sliding window
is cut here — that's only needed later for the continuous Track-A features.

Models (D9)
-----------
  • LSTM       — on the full (T, F) SHARP feature window.
  • Holt-Winters — smooths a single scalar feature series, then a 1-D logistic
    calibrator (fit on TRAIN only) maps its forecast to a probability.
  • Ensemble   — alpha·P_hw + (1-alpha)·P_lstm.

Usage:  python -m src.forecast.train --config configs/lstm_swansf.yaml
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import DATA_DIR, DEVICE, OUT_DIR
from src.data.swan_sf import feature_columns, load_swan_sf
from src.eval.metrics import bss, hss, reliability_table, tss
from src.eval.splits import partition_split
from src.forecast.models import Ensemble, HoltWintersForecaster, LSTMForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy preprocessing (avoids sklearn DLL issues on some Windows setups)
# ---------------------------------------------------------------------------

class _MedianImputer:
    """Fit per-column median on train, replace NaN with it at transform time."""

    def __init__(self):
        self.medians_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_MedianImputer":
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.medians_ is not None, "Call fit() first."
        idx = np.where(np.isnan(X))
        X = X.copy()
        X[idx] = np.take(self.medians_, idx[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _ZScoreScaler:
    """Fit mean/std on train, apply at transform. Constant columns -> 0."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_ZScoreScaler":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first."
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _LogisticCalibrator:
    """1-D logistic regression (numpy) mapping a raw score -> probability.

    Used to turn the Holt-Winters scalar-feature forecast into P(flare).
    Standardises the input score internally; fit on TRAIN labels only (D8).
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = 0.0
        self.b = 0.0
        self.mu_ = 0.0
        self.sd_ = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_LogisticCalibrator":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = float(np.nanmean(x))
        self.sd_ = float(np.nanstd(x)) or 1.0
        xs = (x - self.mu_) / self.sd_
        xs = np.nan_to_num(xs)
        for _ in range(self.n_iter):
            z = self.w * xs + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = float(np.mean((p - y) * xs))
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - self.mu_) / self.sd_
        xs = np.nan_to_num(xs)
        return (1.0 / (1.0 + np.exp(-(self.w * xs + self.b)))).astype(np.float32)


# ---------------------------------------------------------------------------
# Instance -> sample tensors
# ---------------------------------------------------------------------------

def _make_instance_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    hw_feature: str,
    series_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn each SWAN-SF instance into one fixed-length sample.

    Each instance (grouped by ``instance_id``, sorted by ``timestamp``) becomes:
      • feat window  : (series_length, n_features)  — for the LSTM
      • hw series    : (series_length,)             — the scalar HW feature
      • y            : the instance label
      • harpnum      : the instance's AR id (for bookkeeping)

    Series longer than series_length are truncated to the LAST series_length
    steps; shorter ones are front-padded by repeating the first row.

    Returns
    -------
    X        : (N, series_length, n_features) float32
    hw_series: (N, series_length) float32
    y        : (N,) int32
    harp     : (N,) int64
    """
    feats, hws, ys, harps = [], [], [], []
    hw_idx = feature_cols.index(hw_feature) if hw_feature in feature_cols else 0

    for _, g in df.groupby("instance_id", sort=False):
        g = g.sort_values("timestamp")
        fw = g[feature_cols].to_numpy(dtype=np.float32)        # (T, F)
        if len(fw) >= series_length:
            fw = fw[-series_length:]
        else:
            pad = np.repeat(fw[:1], series_length - len(fw), axis=0)
            fw = np.vstack([pad, fw])
        feats.append(fw)
        hws.append(fw[:, hw_idx])
        ys.append(int(g["label"].iloc[0]))
        harps.append(int(g["harpnum"].iloc[0]))

    if not feats:
        n_f = len(feature_cols)
        return (
            np.empty((0, series_length, n_f), np.float32),
            np.empty((0, series_length), np.float32),
            np.empty(0, np.int32),
            np.empty(0, np.int64),
        )
    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(hws, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(harps, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    partitions = cfg.get("swan_sf_partitions", [1, 2, 3, 4, 5])
    train_partitions = cfg.get("train_partitions", [1, 2, 3, 4])
    test_partitions = cfg.get("test_partitions", [5])
    series_length = int(cfg.get("series_length", 60))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # 1. Load -----------------------------------------------------------------
    logger.info("Loading SWAN-SF partitions %s …", partitions)
    df = load_swan_sf(data_dir=DATA_DIR, partitions=partitions)

    feature_cols = feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in SWAN-SF data.")
    hw_feature = cfg.get("hw_feature") or feature_cols[0]
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("HW scalar feature: %s", hw_feature)

    # 2. Partition split (D8: time-block at partition level + whole-AR guard) --
    train_df, test_df = partition_split(df, train_partitions, test_partitions)
    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "Split produced an empty set. Check train_partitions/test_partitions "
            "against the partitions actually loaded."
        )

    # 3. Build samples --------------------------------------------------------
    X_train, hw_train, y_train, _ = _make_instance_samples(
        train_df, feature_cols, hw_feature, series_length
    )
    X_test, hw_test, y_test, _ = _make_instance_samples(
        test_df, feature_cols, hw_feature, series_length
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("No instance samples built — check 'instance_id' grouping.")

    test_base_rate = float(y_test.mean())
    logger.info(
        "Train samples: %d  pos=%.2f%%   |   Test samples: %d  pos=%.2f%% (true rate)",
        len(y_train), 100.0 * y_train.mean(),
        len(y_test), 100.0 * test_base_rate,
    )

    # 4. Resample TRAIN only (D8: test keeps the true base rate) ---------------
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        logger.warning("Train set is single-class — skipping oversampling.")
        bal_idx = np.arange(len(y_train))
    else:
        minority, majority = (pos_idx, neg_idx) if len(pos_idx) <= len(neg_idx) else (neg_idx, pos_idx)
        over = rng.choice(minority, size=len(majority), replace=True)
        bal_idx = np.concatenate([majority, over])
        rng.shuffle(bal_idx)

    X_tr_bal = X_train[bal_idx]
    y_tr_bal = y_train[bal_idx]
    hw_tr_bal = hw_train[bal_idx]
    logger.info(
        "After oversampling: %d samples  pos=%.2f%%",
        len(X_tr_bal), 100.0 * y_tr_bal.mean(),
    )

    # 5. Feature preprocessing — fit on TRAIN, apply to both ------------------
    n_tr, L, n_f = X_tr_bal.shape
    imputer = _MedianImputer().fit(X_tr_bal.reshape(-1, n_f))
    scaler = _ZScoreScaler().fit(imputer.transform(X_tr_bal.reshape(-1, n_f)))

    def _prep(X):
        n = len(X)
        flat = scaler.transform(imputer.transform(X.reshape(-1, n_f)))
        return flat.reshape(n, L, n_f).astype(np.float32)

    X_tr_bal = _prep(X_tr_bal)
    X_test = _prep(X_test)

    # 6. Train LSTM -----------------------------------------------------------
    logger.info("Training LSTM (device=%s) …", DEVICE)
    lstm = LSTMForecaster(
        n_features=n_f,
        hidden=int(cfg.get("lstm_hidden", 64)),
        layers=int(cfg.get("lstm_layers", 2)),
        dropout=float(cfg.get("lstm_dropout", 0.3)),
    )
    lstm.fit(X_tr_bal, y_tr_bal, cfg, device=DEVICE)

    # 7. Holt-Winters + logistic calibration (fit on TRAIN only) -------------
    hw = HoltWintersForecaster(
        alpha=float(cfg.get("hw_smoothing_level", 0.3)),
        beta=float(cfg.get("hw_smoothing_trend", 0.1)),
        clip=False,                       # smoothing a real feature, not labels
    )
    hw_train_fc = hw.predict_batch(hw_tr_bal)
    calibrator = _LogisticCalibrator().fit(hw_train_fc, y_tr_bal)

    # 8. Ensemble -------------------------------------------------------------
    ensemble = Ensemble(
        hw, lstm,
        alpha=float(cfg.get("ensemble_alpha", 0.5)),
        hw_calibrator=calibrator,
    )

    # 9. Evaluate on held-out test — EVALUATED EXACTLY ONCE (D8) -------------
    logger.info("Evaluating on held-out test (once, true base rate) …")
    y_prob = ensemble.predict_proba(
        hw_test, X_test,
        device=DEVICE,
        batch_size=int(cfg.get("lstm_batch_size", 256)),
    )
    y_pred = (y_prob >= 0.5).astype(int)

    tss_val = tss(y_test, y_pred)
    hss_val = hss(y_test, y_pred)
    bss_val = bss(y_test, y_prob)

    print("\n" + "=" * 60)
    print("  Track B - SWAN-SF Results (held-out, true base rate)")
    print("=" * 60)
    print(f"  TSS  = {tss_val:+.4f}")
    print(f"  HSS  = {hss_val:+.4f}")
    print(f"  BSS  = {bss_val:+.4f}")
    print(f"  n_test           = {len(y_test)}")
    print(f"  test_base_rate   = {test_base_rate:.4f}")
    print(f"  pos_pred (>=0.5) = {int(y_pred.sum())} / {len(y_pred)}")
    print("=" * 60)

    rel = reliability_table(y_test, y_prob, n_bins=int(cfg.get("rel_bins", 10)))
    print("\nReliability table:")
    print(rel.to_string(index=False, float_format="%.4f"))
    print()

    # 10. Save metrics --------------------------------------------------------
    results = {
        "tss": float(tss_val),
        "hss": float(hss_val),
        "bss": float(bss_val),
        "n_test": int(len(y_test)),
        "test_base_rate": float(test_base_rate),
        "n_pos_pred": int(y_pred.sum()),
        "train_partitions": list(train_partitions),
        "test_partitions": list(test_partitions),
        "hw_feature": hw_feature,
        "config": {k: v for k, v in cfg.items() if not isinstance(v, (list, dict))},
    }
    out_path = Path(OUT_DIR) / "track_b_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Metrics saved -> %s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Track B: SWAN-SF forecaster (D9)")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    main(cfg)
