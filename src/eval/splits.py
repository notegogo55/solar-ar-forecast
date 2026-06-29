"""Validation splits (D8). The single most important correctness guardrail.

Rules: chronological time-block split; an entire AR (all its windows over its
disk passage) lands in ONLY ONE fold — never straddling. Choose split dates
that do not cut through an AR's passage.
"""


def time_block_whole_ar_split(df, time_col, ar_col, test_window):
    # TODO: assign whole ARs to train/test by time-block; assert no AR_id in both.
    raise NotImplementedError
