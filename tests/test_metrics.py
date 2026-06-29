"""Test suite for src.eval.metrics — written BEFORE implementation (D8 / test-first).

All expected values are hand-computed from the confusion-matrix definitions:
  TSS = TPR - FPR = TP/(TP+FN) - FP/(FP+TN)
  HSS = 2(TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
  BSS = 1 - BS/BS_ref  where BS = mean((prob - label)²)
                              BS_ref = mean((clim - label)²)  clim = mean(label)
"""
import numpy as np
import pytest

from src.eval.metrics import tss, hss, bss, reliability_table


# ---------------------------------------------------------------------------
# TSS
# ---------------------------------------------------------------------------

def test_tss_perfect():
    # TP=2, FN=0, FP=0, TN=2  →  TPR=1, FPR=0  →  TSS=1
    assert tss([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_tss_no_skill():
    # TP=1, FN=1, FP=1, TN=1  →  TPR=0.5, FPR=0.5  →  TSS=0
    assert tss([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_tss_inverse():
    # TP=0, FN=2, FP=2, TN=0  →  TPR=0, FPR=1  →  TSS=-1
    assert tss([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_tss_all_negative_predictions():
    # TP=0, FN=2, FP=0, TN=3  →  TPR=0, FPR=0  →  TSS=0
    assert tss([1, 1, 0, 0, 0], [0, 0, 0, 0, 0]) == pytest.approx(0.0)


def test_tss_all_positive_predictions():
    # TP=2, FN=0, FP=3, TN=0  →  TPR=1, FPR=1  →  TSS=0
    assert tss([1, 1, 0, 0, 0], [1, 1, 1, 1, 1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HSS
# ---------------------------------------------------------------------------

def test_hss_perfect():
    # 2(2·2 - 0·0) / [(2+0)(0+2) + (2+0)(0+2)] = 8/8 = 1
    assert hss([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_hss_below_no_skill():
    # y_true=[1,0,0,0]  y_pred=[0,1,1,1]
    # TP=0, FN=1, FP=3, TN=0
    # num = 2(0·0 - 3·1) = -6
    # den = (0+1)(1+0) + (0+3)(3+0) = 1 + 9 = 10
    # HSS = -6/10 = -0.6
    assert hss([1, 0, 0, 0], [0, 1, 1, 1]) == pytest.approx(-0.6)


def test_hss_all_negative():
    # Predicting all negatives on balanced data gives HSS = 0
    # y_true=[1,0,1,0]  y_pred=[0,0,0,0]
    # TP=0, FN=2, FP=0, TN=2
    # num = 2(0·2 - 0·2) = 0  →  HSS=0
    assert hss([1, 0, 1, 0], [0, 0, 0, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BSS
# ---------------------------------------------------------------------------

def test_bss_perfect_probs():
    # BS=0  →  BSS=1
    assert bss([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_bss_climatology():
    # y_prob = mean(y_true) = 0.5 everywhere  →  BS = BS_ref  →  BSS=0
    assert bss([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_bss_worse_than_clim():
    # Predicting opposite of truth: BSS < 0
    assert bss([1, 0, 1, 0], [0.0, 1.0, 0.0, 1.0]) < 0.0


def test_bss_asymmetric_case():
    # y_true all 1s: clim=1, BS_ref=0  →  BSS defined as 0 (not div-by-zero)
    # This guards the edge case where the test window is all-positive
    result = bss([1, 1, 1, 1], [0.8, 0.9, 0.7, 0.95])
    assert isinstance(result, float)  # must not raise


# ---------------------------------------------------------------------------
# reliability_table
# ---------------------------------------------------------------------------

def test_reliability_table_columns():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_prob = rng.uniform(0, 1, 300)
    df = reliability_table(y_true, y_prob, n_bins=10)
    assert {"bin_center", "mean_pred_prob", "observed_freq", "count"}.issubset(df.columns)


def test_reliability_table_count_sum():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_prob = rng.uniform(0, 1, 300)
    df = reliability_table(y_true, y_prob, n_bins=10)
    assert df["count"].sum() == 300


def test_reliability_table_bin_range():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    df = reliability_table(y_true, y_prob, n_bins=10)
    assert (df["bin_center"] >= 0).all()
    assert (df["bin_center"] <= 1).all()


def test_reliability_table_empty_bins_skipped():
    # All predictions in [0, 0.1] → only 1 bin populated
    y_true = [1, 0, 1, 0, 0]
    y_prob = [0.05, 0.03, 0.08, 0.02, 0.09]
    df = reliability_table(y_true, y_prob, n_bins=10)
    assert len(df) == 1
    assert df["count"].iloc[0] == 5
