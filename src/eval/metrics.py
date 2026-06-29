"""Imbalance-robust metrics (D8). NEVER headline accuracy.

Resampling/weighting happens on TRAIN only; the test set keeps the true base
rate so these numbers don't lie.
"""


def tss(y_true, y_pred):  # True Skill Statistic — primary metric
    raise NotImplementedError


def hss(y_true, y_pred):  # Heidke Skill Score
    raise NotImplementedError


def bss(y_true, y_prob):  # Brier Skill Score
    raise NotImplementedError
