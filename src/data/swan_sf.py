"""SWAN-SF benchmark loader (Angryk et al. 2020, Sci. Data 7, 227).

Dataset: multivariate SHARP-parameter time series for solar flare forecasting.
Canonical source: **Harvard Dataverse** — DOI 10.7910/DVN/EBCFKM
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM

(NOTE: the old bitbucket.org/gsudmlab/swan_sf location was the team's *code*
repo, not the data, and is no longer reachable. Use Dataverse.)

Original distribution format
----------------------------
5 partition archives (partition1 … partition5). Each partition, once extracted,
is a FOLDER of per-instance CSV files — one file per MVTS sample. Each instance:
  • is a fixed-length multivariate time series (≈60 timesteps over a 12 h window
    at 12 min cadence) — i.e. ONE ready-made labelled sample, not a continuous
    AR series that needs sliding-window cutting.
  • columns = the SHARP photospheric parameters + Timestamp (+ optionally
    NOAA_ARS / HARPNUM), one row per timestep.
  • the **flare class label** (max flare class in the prediction window) is
    encoded in the FILENAME prefix, e.g. ``M1.0@...``, ``C3.2@...``, ``B...``,
    ``N...`` / ``FQ...`` for flare-quiet.

The 5 partitions are non-overlapping TIME blocks by design — so the canonical,
D8-compliant protocol is a partition-level split (e.g. train p1–4, test p5),
not an arbitrary timestamp cut.

Expected on-disk layout (download + extract once, gitignored under data/)
------------------------------------------------------------------------
    data/swan_sf/partition1/<instance>.csv
    data/swan_sf/partition2/<instance>.csv
    ...
A flat fallback ``data/swan_sf/partition{N}.csv`` (single tidy file with an
``instance_id`` column) is also accepted — handy for synthetic test fixtures.

Column contract (returned tidy long DataFrame, one row per timestep)
--------------------------------------------------------------------
  instance_id  str       unique id of the MVTS sample (filename stem)
  harpnum      int       HARP number — AR identity for whole-AR guard (D8)
  timestamp    datetime  observation timestamp (T_REC)
  <features>   float     SHARP-parameter columns
  label        int       1 = ≥M1.0 flare in prediction window, else 0  (D7)
  _partition   int       source partition number (1–5)

⚠ VERIFY-ON-REAL-DATA: the exact filename label convention and column names
vary slightly between SWAN-SF releases. The parser auto-detects the CSV
delimiter and matches column names case-insensitively; the label regex and
positive-class set are module constants below — adjust them once you have a
real partition extracted if labels come out wrong (the loader logs the detected
base rate so a mismatch is obvious).
"""
import logging
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_DATAVERSE_DOI = "doi:10.7910/DVN/EBCFKM"
_DATAVERSE_URL = (
    "https://dataverse.harvard.edu/dataset.xhtml?persistentId=" + _DATAVERSE_DOI
)

# Column-name aliases (matched case-insensitively)
_HARPNUM_ALIASES = {"harpnum", "harp_num", "harp", "harpnums"}
_TIME_ALIASES = {"timestamp", "t_rec", "time", "date", "t_obs"}
_NOAA_ALIASES = {"noaa_ars", "noaa_ar", "noaa", "noaa_num"}

# Label parsing from the instance filename.
# Default: leading flare-class token, e.g. "M1.0@..." -> "M1.0" -> class "M".
# ≥M1.0 (D7) means flare class M or X.
_LABEL_REGEX = re.compile(r"^\s*([A-Za-z][0-9.]*)")
_POSITIVE_CLASSES = {"M", "X"}

# Non-feature columns that must never be treated as model inputs.
_META_COLS = {"instance_id", "harpnum", "timestamp", "label", "_partition", "noaa_ars"}


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _partition_dir(data_dir: Path, p: int) -> Path:
    return data_dir / "swan_sf" / f"partition{p}"


def _partition_flat_file(data_dir: Path, p: int) -> Path | None:
    base = data_dir / "swan_sf"
    for ext in (".csv", ".arff"):
        f = base / f"partition{p}{ext}"
        if f.exists():
            return f
    return None


def _missing_partition_error(p: int, data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"SWAN-SF partition {p} not found.\n"
        f"Expected either:\n"
        f"  • a folder of instance CSVs at {_partition_dir(data_dir, p)}/\n"
        f"  • or a flat file {data_dir / 'swan_sf'}/partition{p}.csv\n\n"
        f"Download + extract the dataset from Harvard Dataverse (gitignored):\n"
        f"  {_DATAVERSE_URL}\n"
        f"Dataverse downloads are large zips behind the web UI — fetch manually, "
        f"do not rely on auto-download. See docs/runbook-kaggle.md §9."
    )


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _label_from_filename(stem: str) -> int:
    """Map an instance filename prefix to a binary ≥M1.0 label (D7)."""
    m = _LABEL_REGEX.match(stem)
    cls = m.group(1)[:1].upper() if m else ""
    return 1 if cls in _POSITIVE_CLASSES else 0


def _binarize_label_column(series: pd.Series) -> pd.Series:
    """Fallback: binarize an in-file label column (numeric or nominal)."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    first = series.astype(str).str.strip().str[:1].str.upper()
    return first.isin(_POSITIVE_CLASSES).astype(int)


# ---------------------------------------------------------------------------
# Per-instance / flat readers
# ---------------------------------------------------------------------------

def _read_one_csv(path: Path) -> pd.DataFrame:
    """Read a single CSV, auto-detecting the delimiter (tab or comma)."""
    return pd.read_csv(path, sep=None, engine="python")


def _load_partition_dir(p_dir: Path, partition: int) -> pd.DataFrame:
    """Load every instance CSV in a partition folder into one tidy DataFrame."""
    files = sorted(p_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No *.csv instance files in {p_dir}")
    frames = []
    for f in files:
        df_i = _read_one_csv(f)
        df_i["instance_id"] = f.stem
        df_i["label"] = _label_from_filename(f.stem)
        df_i["_partition"] = partition
        frames.append(df_i)
    logger.info("  partition %d: %d instance files", partition, len(files))
    return pd.concat(frames, ignore_index=True)


def _load_partition_flat(path: Path, partition: int) -> pd.DataFrame:
    """Load a flat single-file partition (test fixtures / pre-tidied data)."""
    df = _read_one_csv(path) if path.suffix == ".csv" else _read_arff(path)
    df["_partition"] = partition
    if "instance_id" not in {c.lower() for c in df.columns}:
        # Synthesize instance ids from (harpnum, contiguous run) if absent:
        # fall back to a per-AR id so downstream grouping still works.
        harp = _find_col(list(df.columns), _HARPNUM_ALIASES)
        if harp:
            df["instance_id"] = (
                f"p{partition}_" + df[harp].astype(str)
            )
    logger.info("  partition %d: flat file %s (%d rows)", partition, path.name, len(df))
    return df


def _read_arff(path: Path) -> pd.DataFrame:
    from scipy.io.arff import loadarff

    data, _ = loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode("utf-8")
            except (AttributeError, UnicodeDecodeError):
                df[col] = df[col].astype(str)
    return df


def _find_col(columns: list[str], aliases: set[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_swan_sf(
    data_dir=None,
    partitions: Sequence[int] = (1, 2, 3, 4, 5),
    download: bool = False,
) -> pd.DataFrame:
    """Load SWAN-SF partitions as a single tidy long DataFrame.

    Parameters
    ----------
    data_dir   : Root data directory (default: src.config.DATA_DIR). Partitions
                 are read from data_dir/swan_sf/partition{N}/ (folder of instance
                 CSVs) or data_dir/swan_sf/partition{N}.csv (flat fallback).
    partitions : Which partition numbers to load (1–5).
    download   : Kept for API symmetry; Dataverse data must be fetched manually
                 (large zips behind the web UI). True still raises with the link.

    Returns
    -------
    Tidy long DataFrame — see the module docstring for the column contract.
    Sorted by (_partition, harpnum, instance_id, timestamp).
    """
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    if download:
        raise RuntimeError(
            "SWAN-SF cannot be auto-downloaded — it lives behind the Harvard "
            f"Dataverse web UI as large zips. Fetch + extract manually to "
            f"{data_dir / 'swan_sf'}/ then call with download=False:\n  {_DATAVERSE_URL}"
        )

    frames = []
    for p in partitions:
        p_dir = _partition_dir(data_dir, p)
        flat = _partition_flat_file(data_dir, p)
        if p_dir.is_dir():
            frames.append(_load_partition_dir(p_dir, p))
        elif flat is not None:
            frames.append(_load_partition_flat(flat, p))
        else:
            raise _missing_partition_error(p, data_dir)

    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns)

    # --- Normalise key columns ---
    rename: dict[str, str] = {}
    harp_raw = _find_col(cols, _HARPNUM_ALIASES)
    if harp_raw:
        rename[harp_raw] = "harpnum"
    time_raw = _find_col(cols, _TIME_ALIASES)
    if time_raw:
        rename[time_raw] = "timestamp"
    noaa_raw = _find_col(cols, _NOAA_ALIASES)
    if noaa_raw:
        rename[noaa_raw] = "noaa_ars"
    df.rename(columns=rename, inplace=True)

    # If no label was set (flat file without filename labels), derive from a column.
    if "label" not in df.columns:
        label_col = _find_col(list(df.columns), {"label", "class", "target"})
        if label_col is None:
            raise RuntimeError(
                "Could not determine a flare label (no filename label and no "
                "'label'/'class' column). Check the SWAN-SF format / loader config."
            )
        df["label"] = _binarize_label_column(df[label_col])

    # Type coercion
    if "harpnum" in df.columns:
        df["harpnum"] = pd.to_numeric(df["harpnum"], errors="coerce").fillna(-1).astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["label"] = df["label"].astype(int)

    # Drop rows missing a critical key
    required = [c for c in ("harpnum", "timestamp", "label") if c in df.columns]
    before = len(df)
    df.dropna(subset=required, inplace=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with NaN in %s", before - len(df), required)

    sort_keys = [
        k for k in ("_partition", "harpnum", "instance_id", "timestamp")
        if k in df.columns
    ]
    df.sort_values(sort_keys, inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_instances = df["instance_id"].nunique() if "instance_id" in df.columns else -1
    logger.info(
        "SWAN-SF loaded: %d rows, %d instances, %d ARs, base_rate=%.4f  partitions=%s",
        len(df),
        n_instances,
        df["harpnum"].nunique() if "harpnum" in df.columns else -1,
        _instance_base_rate(df),
        list(partitions),
    )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric SHARP-parameter feature columns (everything non-meta)."""
    return [
        c for c in df.columns
        if c.lower() not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _instance_base_rate(df: pd.DataFrame) -> float:
    """Positive rate at the INSTANCE level (one label per sample), not per row."""
    if "instance_id" in df.columns:
        per_inst = df.groupby("instance_id")["label"].first()
        return float(per_inst.mean()) if len(per_inst) else float("nan")
    return float(df["label"].mean()) if len(df) else float("nan")