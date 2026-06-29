"""Generate tiny synthetic SWAN-SF-like data for pipeline testing.

Mirrors the ORIGINAL Harvard Dataverse layout so the real loader code path is
exercised end-to-end without the real (large) download:

    data/swan_sf/partition{1-5}/<CLASS>@<idx>_ar<harpnum>.csv

Each instance file = one fixed-length MVTS sample (~60 timesteps), with the
flare class encoded in the filename prefix (M/X => positive, >=M1.0 per D7).
HARPNUM ranges are disjoint per partition (no straddling ARs).

Run:  python scripts/make_mock_swansf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import DATA_DIR

FEATURES = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH",
    "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE",
]
SERIES_LEN = 60          # timesteps per instance (12 h @ 12 min)
ARS_PER_PARTITION = 60
INSTANCES_PER_AR = 5     # => 300 instances/partition
FLARE_RATE = 0.08        # ~8 % positive (M/X)

PARTITION_STARTS = {
    1: "2010-06-01", 2: "2011-06-01", 3: "2012-06-01",
    4: "2013-06-01", 5: "2014-06-01",
}


def _class_for(is_pos: bool, rng: np.random.Generator) -> str:
    if is_pos:
        return rng.choice(["M", "X"], p=[0.85, 0.15])
    return rng.choice(["N", "B", "C"], p=[0.5, 0.3, 0.2])


def make_partition(partition: int) -> int:
    rng = np.random.default_rng(partition * 17)
    p_dir = DATA_DIR / "swan_sf" / f"partition{partition}"
    p_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(PARTITION_STARTS[partition])
    harp_base = 1000 + partition * 100      # disjoint HARPNUM range per partition

    n = 0
    for ar_offset in range(ARS_PER_PARTITION):
        harpnum = harp_base + ar_offset
        for inst in range(INSTANCES_PER_AR):
            is_pos = rng.random() < FLARE_RATE
            cls = _class_for(is_pos, rng)
            # Slightly shift positive-instance feature means so a signal exists.
            shift = 0.6 if is_pos else 0.0
            ts0 = start + pd.Timedelta(days=ar_offset, hours=inst * 13)
            rows = []
            for t in range(SERIES_LEN):
                vals = rng.standard_normal(len(FEATURES)) + shift
                row = {
                    "Timestamp": (ts0 + pd.Timedelta(minutes=12 * t)).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "HARPNUM": harpnum,
                }
                row.update(dict(zip(FEATURES, vals)))
                rows.append(row)
            fname = f"{cls}@{n}_ar{harpnum}.csv"
            pd.DataFrame(rows).to_csv(p_dir / fname, index=False)
            n += 1
    print(f"  partition {partition}: {n} instance files -> {p_dir}")
    return n


def main() -> None:
    total = 0
    for p in range(1, 6):
        p_dir = DATA_DIR / "swan_sf" / f"partition{p}"
        if p_dir.is_dir() and any(p_dir.glob("*.csv")):
            print(f"  partition {p}: already populated -> {p_dir} (skipped)")
            continue
        total += make_partition(p)
    print(f"\nDone ({total} instances). Run the forecaster with:")
    print("  python -m src.forecast.train --config configs/lstm_swansf.yaml")


if __name__ == "__main__":
    main()
