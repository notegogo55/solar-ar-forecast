"""Validation splits (D8). The single most important correctness guardrail.

Rules: chronological time-block split; an entire AR (all its windows over its
disk passage) lands in ONLY ONE fold — never straddling. Choose split dates
that do not cut through an AR's passage.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def time_block_whole_ar_split(
    df: pd.DataFrame,
    time_col: str,
    ar_col: str,
    test_start,
    test_end=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological time-block split that respects whole-AR passages (D8).

    Assignment rules
    ----------------
    • AR whose last window is strictly before test_start  → **train**
    • AR whose first window is >= test_start              → **test**
    • AR that straddles the boundary                      → **dropped** (logged)

    Parameters
    ----------
    df         : DataFrame with at least time_col and ar_col.
    time_col   : Name of the datetime column.
    ar_col     : Name of the AR identity column (e.g. HARPNUM).
    test_start : Start of the test window (str or Timestamp).
    test_end   : Optional end of the test window (None = no upper bound).

    Returns
    -------
    (train_df, test_df)  — both reset to a fresh integer index.

    Raises
    ------
    AssertionError if any AR ID appears in both returned sets (should never
    happen, but guards against future bugs).
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    test_start = pd.Timestamp(test_start)

    # Per-AR temporal extent
    ar_stats = df.groupby(ar_col).agg(
        first=(time_col, "min"),
        last=(time_col, "max"),
    )

    train_ars: list = []
    test_ars: list = []
    dropped: list = []

    for ar_id, row in ar_stats.iterrows():
        before_cutoff = row["last"] < test_start
        after_cutoff = row["first"] >= test_start
        if before_cutoff:
            train_ars.append(ar_id)
        elif after_cutoff:
            test_ars.append(ar_id)
        else:
            dropped.append(ar_id)

    if dropped:
        logger.warning(
            "time_block_whole_ar_split: %d AR(s) straddle test_start=%s and are dropped "
            "from both sets: %s%s",
            len(dropped),
            test_start.date(),
            dropped[:10],
            " …" if len(dropped) > 10 else "",
        )

    train_df = df[df[ar_col].isin(train_ars)].reset_index(drop=True)
    test_df = df[df[ar_col].isin(test_ars)].reset_index(drop=True)

    # Optionally truncate test to test_end
    if test_end is not None:
        test_end_ts = pd.Timestamp(test_end)
        test_df = test_df[test_df[time_col] <= test_end_ts].reset_index(drop=True)

    # Invariant: no AR can appear in both sets
    overlap = set(train_df[ar_col]) & set(test_df[ar_col])
    assert len(overlap) == 0, (
        f"BUG: {len(overlap)} AR(s) appear in both train and test sets: "
        f"{sorted(overlap)[:10]}"
    )

    logger.info(
        "Split done — train: %d rows / %d ARs  |  test: %d rows / %d ARs  "
        "|  dropped ARs: %d",
        len(train_df), len(train_ars),
        len(test_df), len(test_ars),
        len(dropped),
    )
    return train_df, test_df


def partition_split(
    df: pd.DataFrame,
    train_partitions: list[int],
    test_partitions: list[int],
    partition_col: str = "_partition",
    ar_col: str = "harpnum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition-level split for SWAN-SF — the canonical D8 protocol.

    SWAN-SF's 5 partitions are non-overlapping TIME blocks by design, so
    assigning whole partitions to train vs. test already satisfies the
    time-block requirement. This adds the whole-AR guard on top: any AR
    (HARPNUM) that appears in BOTH the chosen train and test partitions is
    dropped from the test set, so no AR straddles the fold boundary.

    Parameters
    ----------
    df               : Tidy DataFrame with a partition column and an AR column.
    train_partitions : Partition numbers used for training (e.g. [1, 2, 3, 4]).
    test_partitions  : Partition numbers used for the held-out test (e.g. [5]).
    partition_col    : Name of the partition column.
    ar_col           : Name of the AR identity column.

    Returns
    -------
    (train_df, test_df) — reset to a fresh integer index.
    """
    overlap_p = set(train_partitions) & set(test_partitions)
    assert not overlap_p, f"train/test partitions overlap: {overlap_p}"

    train_df = df[df[partition_col].isin(train_partitions)].reset_index(drop=True)
    test_df = df[df[partition_col].isin(test_partitions)].reset_index(drop=True)

    # Whole-AR guard: drop from TEST any AR also present in TRAIN (never the
    # reverse — we protect the test set's independence).
    straddlers = set(train_df[ar_col]) & set(test_df[ar_col])
    if straddlers:
        logger.warning(
            "partition_split: %d AR(s) appear in both train and test partitions; "
            "dropping them from TEST to keep folds AR-disjoint: %s%s",
            len(straddlers),
            sorted(straddlers)[:10],
            " …" if len(straddlers) > 10 else "",
        )
        test_df = test_df[~test_df[ar_col].isin(straddlers)].reset_index(drop=True)

    overlap = set(train_df[ar_col]) & set(test_df[ar_col])
    assert len(overlap) == 0, f"BUG: AR(s) still in both sets: {sorted(overlap)[:10]}"

    logger.info(
        "Partition split — train p%s: %d rows / %d ARs  |  test p%s: %d rows / %d ARs",
        train_partitions, len(train_df), train_df[ar_col].nunique(),
        test_partitions, len(test_df), test_df[ar_col].nunique(),
    )
    return train_df, test_df
