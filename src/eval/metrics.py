"""Imbalance-robust metrics (D8). NEVER headline accuracy.

Resampling/weighting happens on TRAIN only; the test set keeps the true base
rate so these numbers don't lie.

All functions accept plain Python lists or numpy arrays of ints/floats.
"""
import numpy as np
import pandas as pd


def _cm(y_true, y_pred):
    """Return (TP, FN, FP, TN) from binary classification arrays."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    return TP, FN, FP, TN


def tss(y_true, y_pred) -> float:
    """True Skill Statistic = TPR - FPR.

    Range [-1, 1].  0 = no skill (random or constant predictor).
    Primary metric (D8) because it is independent of class prevalence.
    """
    TP, FN, FP, TN = _cm(y_true, y_pred)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return float(tpr - fpr)


def hss(y_true, y_pred) -> float:
    """Heidke Skill Score.

    Range (-inf, 1].  0 = no skill.  Negative = worse than random chance.
    """
    TP, FN, FP, TN = _cm(y_true, y_pred)
    num = 2 * (TP * TN - FP * FN)
    den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    return float(num / den) if den > 0 else 0.0


def bss(y_true, y_prob) -> float:
    """Brier Skill Score = 1 - BS / BS_ref.

    BS_ref is the Brier score of the climatology (constant) forecast.
    Range (-inf, 1].  0 = no improvement over climatology.  1 = perfect.
    Returns 0.0 if BS_ref == 0 (all labels identical — undefined but safe).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bs = float(np.mean((y_prob - y_true) ** 2))
    clim = float(y_true.mean())
    bs_ref = float(np.mean((clim - y_true) ** 2))
    if bs_ref == 0.0:
        return 0.0
    return float(1.0 - bs / bs_ref)


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Calibration / reliability table.

    Returns a DataFrame with columns:
        bin_center      float   midpoint of the probability bin
        mean_pred_prob  float   mean predicted probability within the bin
        observed_freq   float   fraction of positives within the bin
        count           int     number of samples in the bin

    Empty bins are omitted (no zero-count rows).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # Include the right edge in the last bin so prob==1.0 is captured
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        rows.append({
            "bin_center": float((lo + hi) / 2),
            "mean_pred_prob": float(y_prob[mask].mean()),
            "observed_freq": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)
