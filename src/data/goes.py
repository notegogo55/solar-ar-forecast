"""GOES X-ray flux and NOAA/SWPC flare event list.

GOES flux  -> feature input (1-8 A proxy for solar activity level).
Flare list -> label source: which NOAA AR produced which >=M1.0 flare (D7).

Data sources (no login required, public):
  Events : https://www.swpc.noaa.gov/pub/indices/events/{YYYYMM}events.txt
  XRS    : sunpy Fido (GOES data client) -> NetCDF4 via NOAA NCEI
"""
import logging
import urllib.request
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_SWPC_EVENTS_URL = (
    "https://www.swpc.noaa.gov/pub/indices/events/{year}{month:02d}events.txt"
)
# Letter order: A=1e-8, B=1e-7, C=1e-6, M=1e-5, X=1e-4 W/m^2
_CLASS_RANK = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


# ---------------------------------------------------------------------------
# NOAA flare event list
# ---------------------------------------------------------------------------

def fetch_noaa_events(start: str, end: str) -> pd.DataFrame:
    """Download NOAA/SWPC flare event list for [start, end].

    Tries SWPC HTTP archive first; falls back to sunpy HEK if that fails.
    Returns DataFrame: start_time, peak_time, end_time, noaa_ar (int),
    goes_class (str), is_ge_M1 (bool).
    """
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")

    # --- Primary: SWPC monthly event text files ---
    dfs = []
    for year, month in _months_in_range(t0, t1):
        url = _SWPC_EVENTS_URL.format(year=year, month=month)
        log.info("NOAA events (SWPC): %s", url)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            df = _parse_swpc_events(text)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            log.warning("SWPC fetch failed: %s", exc)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df[(df["start_time"] >= t0) & (df["start_time"] <= t1)].copy()
        log.info("NOAA events: %d total, %d >=M1.0", len(df), df["is_ge_M1"].sum())
        return df

    # --- Fallback: sunpy HEK (Heliophysics Event Knowledgebase) ---
    log.warning("SWPC archive unavailable -- trying sunpy HEK fallback")
    return _fetch_hek_events(start, end)


def _fetch_hek_events(start: str, end: str) -> pd.DataFrame:
    """Fetch flare events from sunpy HEK as fallback for SWPC archive."""
    try:
        from sunpy.net import Fido, attrs as a
        log.info("HEK query: FL events %s -> %s", start, end)
        result = Fido.search(
            a.Time(start, end),
            a.hek.EventType("FL"),
        )
        table = Fido.fetch(result[0])   # result[0] is the HEK response table
        # HEK returns an astropy Table or pandas-like object
        rows = []
        for row in table:
            goes_cls = str(row.get("fl_goescls", "") or "")
            if not goes_cls or goes_cls[0] not in ("C", "M", "X"):
                continue
            try:
                t_start = pd.Timestamp(str(row["event_starttime"]), tz="UTC")
                t_peak = pd.Timestamp(str(row["event_peaktime"]), tz="UTC")
                t_end = pd.Timestamp(str(row["event_endtime"]), tz="UTC")
            except Exception:
                continue
            ar_raw = str(row.get("ar_noaanum", "") or "0")
            try:
                noaa_ar = int(ar_raw)
            except ValueError:
                noaa_ar = 0
            rows.append({
                "start_time": t_start,
                "peak_time": t_peak,
                "end_time": t_end,
                "goes_class": goes_cls,
                "noaa_ar": noaa_ar,
                "is_ge_M1": _class_ge_M1(goes_cls),
            })
        df = pd.DataFrame(rows) if rows else _empty_events()
        log.info("HEK events: %d total, %d >=M1.0", len(df), df["is_ge_M1"].sum())
        return df
    except Exception as exc:
        log.warning("HEK fallback also failed: %s", exc)
        return _empty_events()


def fetch_goes_flux(start: str, end: str) -> pd.DataFrame:
    """Download 1-minute GOES X-ray flux via sunpy Fido.

    Returns DataFrame: time (UTC Timestamps), xrsa (1-8 A W/m^2),
    xrsb (0.5-4 A W/m^2). Returns empty DataFrame on failure so that
    label construction (which uses only the NOAA event list) is unaffected.
    """
    try:
        from sunpy.net import Fido, attrs as a
    except ImportError:
        log.warning("sunpy.net not available; skipping GOES flux download")
        return _empty_flux()

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    # GOES-15 operated through 2020; GOES-16 from 2017 onward
    sat = 15 if t0.year < 2020 else 16
    log.info("GOES-%d XRS flux %s -> %s", sat, start, end)

    try:
        result = Fido.search(
            a.Time(t0.isoformat(), t1.isoformat()),
            a.Instrument("XRS"),
            a.goes.SatelliteNumber(sat),
        )
        if not result:
            log.warning("No GOES XRS data found")
            return _empty_flux()
        files = Fido.fetch(result, progress=False)
        return _parse_goes_files(files)
    except Exception as exc:
        log.warning("GOES flux download failed: %s", exc)
        return _empty_flux()


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_swpc_events(text: str) -> pd.DataFrame:
    """Parse one NOAA SWPC monthly events text file.

    Format (fixed columns, space-delimited):
      YYYY MM DD  HHMM  HHMM  HHMM  Class  Location  NOAARegion ...
    Comment lines start with # or :.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in ("#", ":"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        # Require first three tokens to be integers (year, month, day)
        try:
            yr, mon, day = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if not (1990 <= yr <= 2100 and 1 <= mon <= 12 and 1 <= day <= 31):
            continue

        t_start = _parse_hhmm(yr, mon, day, parts[3])
        t_peak = _parse_hhmm(yr, mon, day, parts[4])
        t_end = _parse_hhmm(yr, mon, day, parts[5])
        if pd.isnull(t_start):
            continue

        goes_class = parts[6]
        letter = goes_class[0].upper() if goes_class else ""
        if letter not in _CLASS_RANK:
            continue   # skip non-flare event types

        # NOAA AR number: typically 5-digit integer in positions 8+
        noaa_ar = 0
        for tok in parts[7:]:
            if tok.isdigit() and len(tok) == 5:
                noaa_ar = int(tok)
                break

        rows.append({
            "start_time": t_start,
            "peak_time": t_peak,
            "end_time": t_end,
            "goes_class": goes_class,
            "noaa_ar": noaa_ar,
            "is_ge_M1": _class_ge_M1(goes_class),
        })

    return pd.DataFrame(rows) if rows else _empty_events()


def _parse_goes_files(files) -> pd.DataFrame:
    """Parse sunpy-fetched GOES XRS NetCDF4 files via xarray."""
    try:
        import xarray as xr
        dfs = []
        for f in files:
            ds = xr.open_dataset(f)
            time = pd.to_datetime(ds["time"].values, utc=True)
            # Variable names differ between GOES-15 and GOES-16 products.
            # Must use explicit 'is not None' — Python 'or' crashes on DataArrays.
            xrsa = next(
                (ds.get(k) for k in ("xrsa_flux", "a_flux", "xrsa", "A_FLUX")
                 if ds.get(k) is not None), None
            )
            xrsb = next(
                (ds.get(k) for k in ("xrsb_flux", "b_flux", "xrsb", "B_FLUX")
                 if ds.get(k) is not None), None
            )
            dfs.append(pd.DataFrame({
                "time": time,
                "xrsa": xrsa.values.ravel() if xrsa is not None else float("nan"),
                "xrsb": xrsb.values.ravel() if xrsb is not None else float("nan"),
            }))
        return (
            pd.concat(dfs, ignore_index=True)
            .sort_values("time")
            .reset_index(drop=True)
        )
    except Exception as exc:
        log.warning("GOES file parse failed: %s", exc)
        return _empty_flux()


def _class_ge_M1(goes_class: str) -> bool:
    """Return True if goes_class >= M1.0."""
    if not goes_class or len(goes_class) < 2:
        return False
    letter = goes_class[0].upper()
    rank = _CLASS_RANK.get(letter, -1)
    if rank < _CLASS_RANK["M"]:
        return False
    if rank > _CLASS_RANK["M"]:
        return True
    # Letter is M: check mantissa
    try:
        return float(goes_class[1:]) >= 1.0
    except ValueError:
        return False


def _parse_hhmm(year: int, month: int, day: int, hhmm: str):
    hhmm = hhmm.strip()
    if len(hhmm) != 4 or not hhmm.isdigit():
        return pd.NaT
    h, m = int(hhmm[:2]), int(hhmm[2:])
    # Handle 2400 (midnight rollover)
    if h == 24:
        h, day = 0, day + 1
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=h, minute=m, tz="UTC")
    except Exception:
        return pd.NaT


def _months_in_range(t0: pd.Timestamp, t1: pd.Timestamp) -> list:
    months = []
    cur = pd.Timestamp(year=t0.year, month=t0.month, day=1)
    end = pd.Timestamp(year=t1.year, month=t1.month, day=1)
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "start_time", "peak_time", "end_time", "goes_class", "noaa_ar", "is_ge_M1",
    ])


def _empty_flux() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "xrsa", "xrsb"])
