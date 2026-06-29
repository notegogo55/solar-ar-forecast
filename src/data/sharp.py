"""SHARP (Spaceweather HMI Active Region Patches) metadata queries.

Fetches per-AR keyword table from JSOC via drms — tabular metadata only,
no image downloads. Used to build the AR identity table and apply the
per-timestep filters specified in configs/data_v1.yaml.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

SHARP_SERIES = "hmi.sharp_cea_720s"
SHARP_KEYS = [
    "HARPNUM", "T_REC", "LON_FWT", "LAT_FWT",
    "USFLUX", "AREA_ACR", "QUALITY", "NOAA_ARS",
]


def fetch_sharp_metadata(client, start: str, end: str) -> pd.DataFrame:
    """Query SHARP keyword table for the interval [start, end].

    Returns DataFrame with lowercase columns: harpnum, t_rec, lon_fwt,
    lat_fwt, usflux, area_acr, quality, noaa_ars.
    t_rec is UTC-aware Timestamps; harpnum and quality are int.
    """
    t0 = _jsoc_time(start)
    duration = _duration_str(start, end)
    rec_query = f"{SHARP_SERIES}[][{t0}/{duration}@720s]"
    log.info("SHARP query: %s", rec_query)

    keys = client.query(rec_query, key=SHARP_KEYS)
    if keys is None or len(keys) == 0:
        log.warning("No SHARP records for %s → %s", start, end)
        return pd.DataFrame(columns=[k.lower() for k in SHARP_KEYS])

    df = keys.rename(columns={k: k.lower() for k in SHARP_KEYS})
    df["t_rec"] = pd.to_datetime(df["t_rec"], utc=True)
    for col in ("harpnum", "quality"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("lon_fwt", "lat_fwt", "usflux", "area_acr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("SHARP: %d records, %d unique HARPs", len(df), df["harpnum"].nunique())
    return df


def parse_noaa_ars(noaa_ars_str) -> list:
    """Parse NOAA_ARS keyword string (e.g. '12192 12193') → list[int]."""
    s = str(noaa_ars_str).strip()
    if s in ("", "MISSING", "nan", "None"):
        return []
    return [int(x) for x in s.split() if x.isdigit()]


def apply_timestep_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply per-timestep filters from cfg['filters']:
    - |lon_fwt| <= central_meridian_deg  (limb exclusion, D5)
    - quality == 0 if drop_bad_quality   (SHARP QUALITY keyword)
    Returns filtered copy.
    """
    flt = cfg.get("filters", {})
    n0 = len(df)

    lon_limit = flt.get("central_meridian_deg", 70)
    df = df[df["lon_fwt"].abs() <= lon_limit].copy()
    log.info("lon filter (|lon| <= %d°): %d → %d rows", lon_limit, n0, len(df))

    if flt.get("drop_bad_quality", True):
        n1 = len(df)
        df = df[df["quality"] == 0].copy()
        log.info("quality filter: %d → %d rows", n1, len(df))

    return df


def apply_ar_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply per-AR filters (operate on the surviving timestep rows):
    - require_noaa_number: drop HARPs with no NOAA AR number (D4)
    - min_coverage_hours: drop HARPs with fewer than N valid frames
    Returns filtered copy (all rows from HARPs that pass both filters).
    """
    flt = cfg.get("filters", {})

    if flt.get("require_noaa_number", True):
        has_noaa = df["noaa_ars"].apply(lambda x: len(parse_noaa_ars(x)) > 0)
        n_before = df["harpnum"].nunique()
        df = df[has_noaa].copy()
        log.info(
            "require_noaa_number: %d → %d HARPs",
            n_before, df["harpnum"].nunique(),
        )

    min_h = flt.get("min_coverage_hours", 48)
    cadence_h = 720 / 3600          # 720 s per SHARP record = 0.2 h
    min_frames = int(min_h / cadence_h)
    counts = df.groupby("harpnum").size()
    valid_harps = counts[counts >= min_frames].index
    n_before = df["harpnum"].nunique()
    df = df[df["harpnum"].isin(valid_harps)].copy()
    log.info(
        "min_coverage %dh (%d frames): %d → %d HARPs",
        min_h, min_frames, n_before, df["harpnum"].nunique(),
    )

    return df


def _jsoc_time(t: str) -> str:
    return pd.Timestamp(t).strftime("%Y.%m.%d_%H:%M:%S_TAI")


def _duration_str(start: str, end: str) -> str:
    delta = pd.Timestamp(end) - pd.Timestamp(start)
    hours = int(delta.total_seconds() / 3600)
    return f"{hours}h"
