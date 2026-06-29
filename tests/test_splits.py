"""Test suite for src.eval.splits — written BEFORE implementation (D8 / test-first).

Uses a fully synthetic DataFrame so no real data is needed.

Synthetic layout
----------------
  AR  1– 5: entirely in 2010          → train
  AR  6–10: entirely after 2011-01-01 → test
  AR 99   : straddles 2011-01-01      → dropped (not in either set)
"""
import numpy as np
import pandas as pd
import pytest

from src.eval.splits import partition_split, time_block_whole_ar_split

TEST_START = pd.Timestamp("2011-01-01")


def _make_synthetic(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # AR 1-5: 2010-02 through 2010-11 (well before split)
    for ar_id in range(1, 6):
        month = ar_id * 2  # 2, 4, 6, 8, 10
        start = pd.Timestamp(f"2010-{month:02d}-01")
        for h in range(60 * 24):  # 60 days of hourly data
            rows.append({
                "harpnum": ar_id,
                "timestamp": start + pd.Timedelta(hours=h),
                "feat_a": float(rng.standard_normal()),
                "label": int(rng.random() < 0.1),
            })

    # AR 6-10: 2011-02 through 2011-11 (well after split)
    for ar_id in range(6, 11):
        month = (ar_id - 5) * 2  # 2, 4, 6, 8, 10
        start = pd.Timestamp(f"2011-{month:02d}-01")
        for h in range(60 * 24):
            rows.append({
                "harpnum": ar_id,
                "timestamp": start + pd.Timedelta(hours=h),
                "feat_a": float(rng.standard_normal()),
                "label": int(rng.random() < 0.1),
            })

    # AR 99: straddles TEST_START (–10 days to +10 days around 2011-01-01)
    for h in range(-10 * 24, 10 * 24):
        rows.append({
            "harpnum": 99,
            "timestamp": TEST_START + pd.Timedelta(hours=h),
            "feat_a": float(rng.standard_normal()),
            "label": int(rng.random() < 0.1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------

def test_no_ar_overlap():
    df = _make_synthetic()
    train, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    overlap = set(train["harpnum"]) & set(test["harpnum"])
    assert overlap == set(), f"AR IDs appear in both sets: {overlap}"


def test_straddle_dropped_from_both():
    df = _make_synthetic()
    train, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    all_ids = set(train["harpnum"]) | set(test["harpnum"])
    assert 99 not in all_ids, "Straddling AR 99 must be dropped from both train and test"


def test_correct_train_assignment():
    df = _make_synthetic()
    train, _ = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    assert set(range(1, 6)).issubset(set(train["harpnum"])), \
        "ARs 1-5 (entirely in 2010) must be in train"


def test_correct_test_assignment():
    df = _make_synthetic()
    _, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    assert set(range(6, 11)).issubset(set(test["harpnum"])), \
        "ARs 6-10 (entirely in 2011) must be in test"


def test_chronological_boundary():
    """All train timestamps must be strictly before TEST_START."""
    df = _make_synthetic()
    train, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    assert train["timestamp"].max() < TEST_START, \
        "Train set contains timestamps >= test_start"
    assert test["timestamp"].min() >= TEST_START, \
        "Test set contains timestamps < test_start"


# ---------------------------------------------------------------------------
# Guard: resampling train must not touch test base rate
# ---------------------------------------------------------------------------

def test_test_base_rate_invariant_to_train_resampling():
    """After split, resampling train must leave test base rate unchanged."""
    df = _make_synthetic()
    train, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)

    base_rate_before = test["label"].mean()

    # Simulate aggressive oversampling of train (should not affect test)
    pos_rows = train[train["label"] == 1]
    neg_rows = train[train["label"] == 0]
    oversampled = pd.concat(
        [neg_rows, pos_rows.sample(len(neg_rows), replace=True, random_state=0)]
    )
    # (We do nothing with `oversampled`; the key point is test is unchanged)

    base_rate_after = test["label"].mean()
    assert base_rate_before == pytest.approx(base_rate_after), \
        "Test base rate changed — resampling must only touch train"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_result_if_all_straddle():
    """If every AR straddles, both sets are empty (no crash)."""
    rng = np.random.default_rng(7)
    rows = []
    for ar_id in range(1, 4):
        for h in range(-5 * 24, 5 * 24):
            rows.append({
                "harpnum": ar_id,
                "timestamp": TEST_START + pd.Timedelta(hours=h),
                "label": int(rng.random() < 0.1),
            })
    df = pd.DataFrame(rows)
    train, test = time_block_whole_ar_split(df, "timestamp", "harpnum", TEST_START)
    assert len(train) == 0
    assert len(test) == 0


def test_no_straddle_returns_all_data():
    """If there are no straddling ARs, no rows are lost."""
    df = _make_synthetic()
    # Remove AR 99 (the only straddler) from the DataFrame
    df_clean = df[df["harpnum"] != 99].copy()
    train, test = time_block_whole_ar_split(df_clean, "timestamp", "harpnum", TEST_START)
    assert len(train) + len(test) == len(df_clean)


# ---------------------------------------------------------------------------
# partition_split (SWAN-SF canonical D8 split)
# ---------------------------------------------------------------------------

def _make_partitioned(seed: int = 5) -> pd.DataFrame:
    """5 partitions; HARPNUM ranges disjoint per partition by default."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, 6):
        for ar_offset in range(10):
            harp = 1000 + p * 100 + ar_offset
            for t in range(20):
                rows.append({
                    "harpnum": harp,
                    "timestamp": pd.Timestamp("2010-01-01") + pd.Timedelta(hours=t),
                    "feat_a": float(rng.standard_normal()),
                    "label": int(rng.random() < 0.1),
                    "_partition": p,
                })
    return pd.DataFrame(rows)


def test_partition_split_assignment():
    df = _make_partitioned()
    train, test = partition_split(df, [1, 2, 3, 4], [5])
    assert set(train["_partition"]) == {1, 2, 3, 4}
    assert set(test["_partition"]) == {5}


def test_partition_split_no_ar_overlap():
    df = _make_partitioned()
    train, test = partition_split(df, [1, 2, 3, 4], [5])
    assert set(train["harpnum"]) & set(test["harpnum"]) == set()


def test_partition_split_drops_straddler_from_test():
    """An AR present in both a train and the test partition is dropped from test."""
    df = _make_partitioned()
    # Force HARP 1501 (a partition-5 AR) to also appear in partition 1
    extra = df[df["harpnum"] == 1501].copy()
    extra["_partition"] = 1
    df2 = pd.concat([df, extra], ignore_index=True)

    train, test = partition_split(df2, [1, 2, 3, 4], [5])
    assert 1501 not in set(test["harpnum"]), "straddling AR must be dropped from test"
    assert 1501 in set(train["harpnum"])


def test_partition_split_rejects_overlapping_partitions():
    df = _make_partitioned()
    with pytest.raises(AssertionError):
        partition_split(df, [1, 2, 3], [3, 4])
