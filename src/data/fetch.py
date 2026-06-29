"""Track A / A1 -- full-disk 1024^2 event-window acquisition via drms/JSOC.

Designed to run on a Kaggle dataprep kernel (internet ON, GPU off).
See docs/runbook-kaggle.md sec 3 for the full workflow.

Usage:
  python -m src.data.fetch --config configs/data_v1.yaml            # full fetch
  python -m src.data.fetch --config configs/data_v1.yaml --dry-run  # 1 day + 1 channel
  python -m src.data.fetch --config configs/data_v1.yaml --mock     # offline dev/test

D5 constraint: all rescaling happens server-side (JSOC im_patch op=rescale).
Never downsample locally -- the shape assertion in _validate_resolution enforces this.
"""
import argparse
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.config import OUT_DIR
from src.data.goes import fetch_goes_flux, fetch_noaa_events
from src.data.sharp import (
    apply_ar_filters,
    apply_timestep_filters,
    fetch_sharp_metadata,
    parse_noaa_ars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# JSOC series identifiers
_AIA_SERIES = "aia.lev1_euv_12s"
_HMI_SERIES = "hmi.M_720s"
# Native full-disk resolution for both instruments
_AIA_NATIVE = 4096
_HMI_NATIVE = 4096


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path) -> dict:
    """Load and lightly validate configs/data_v1.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _check_required(cfg, [
        "resolution", "cadence_minutes", "channels_aia",
        "include_hmi_magnetogram", "event_windows", "fetch_order",
        "filters", "predict_window_hours", "sample_stride_hours",
    ])
    assert 0 < cfg["resolution"] <= 4096, "resolution must be in (0, 4096]"
    assert cfg["cadence_minutes"] > 0
    return cfg


def _check_required(cfg: dict, keys: list) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")


# ---------------------------------------------------------------------------
# SHARP / AR identity
# ---------------------------------------------------------------------------

def build_sharp_query(client, window: dict, cfg: dict) -> pd.DataFrame:
    """Fetch SHARP metadata for one event window and apply all filters."""
    df = fetch_sharp_metadata(client, window["start"], window["end"])
    if df.empty:
        return df
    df = apply_timestep_filters(df, cfg)
    df = apply_ar_filters(df, cfg)
    return df


def build_noaa_map(sharp_df: pd.DataFrame) -> dict:
    """Return {harpnum -> first_noaa_ar_int} mapping from SHARP data."""
    noaa_map: dict = {}
    for harpnum, grp in sharp_df.groupby("harpnum"):
        for val in grp["noaa_ars"]:
            ars = parse_noaa_ars(val)
            if ars:
                noaa_map[int(harpnum)] = ars[0]
                break
        if int(harpnum) not in noaa_map:
            noaa_map[int(harpnum)] = 0
    return noaa_map


# ---------------------------------------------------------------------------
# Image export
# ---------------------------------------------------------------------------

def _jsoc_time(t) -> str:
    return pd.Timestamp(t).strftime("%Y.%m.%d_%H:%M:%S_TAI")


def _all_channels(cfg: dict) -> list:
    channels = [f"AIA_{w}" for w in cfg.get("channels_aia", [])]
    if cfg.get("include_hmi_magnetogram", True):
        channels.append("HMI_M")
    return channels


def _build_queries(start: str, end: str, channels_aia: list,
                   include_hmi: bool, cadence_min: int) -> list:
    """Return list of {series, query, channel} dicts for JSOC export."""
    t0 = _jsoc_time(start)
    duration_h = int(
        (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600
    ) + 1
    queries = []
    for wav in channels_aia:
        queries.append({
            "series": _AIA_SERIES,
            "query": f"{_AIA_SERIES}[{t0}/{duration_h}h@{cadence_min}m][{wav}]",
            "channel": f"AIA_{wav}",
        })
    if include_hmi:
        queries.append({
            "series": _HMI_SERIES,
            "query": f"{_HMI_SERIES}[{t0}/{duration_h}h@{cadence_min}m]",
            "channel": "HMI_M",
        })
    return queries


def request_image_export(client, queries: list, resolution: int,
                          jsoc_email: str) -> list:
    """Submit async JSOC export requests with server-side rescale (D5).

    Uses im_patch op=rescale so images arrive at `resolution`x`resolution`
    without any local downsampling. Returns list of
    {req: ExportRequest, channel: str} dicts.
    """
    aia_scale = resolution / _AIA_NATIVE
    hmi_scale = resolution / _HMI_NATIVE
    export_reqs = []
    for q in queries:
        scale = aia_scale if q["series"] == _AIA_SERIES else hmi_scale
        log.info("Export: %s  (scale=%.4f)", q["query"], scale)
        req = client.export(
            q["query"],
            email=jsoc_email,
            method="url",
            protocol="fits",
            # Server-side rescale: op=rescale scales the full-disk image
            # to `resolution`x`resolution` before download (D5 constraint).
            process={"im_patch": {"op": "rescale", "scale": scale,
                                   "do_stretchmarks": 0}},
        )
        log.info("  request_id=%s  status=%s", req.id, req.status)
        export_reqs.append({"req": req, "channel": q["channel"]})
    return export_reqs


def poll_and_download(export_reqs: list, out_dir: Path,
                      max_wait_s: int = 7200) -> list:
    """Poll JSOC until all exports complete, then download FITS files.

    Returns list of {channel, timestamp, file_path} dicts.
    Raises TimeoutError if exports are still pending after max_wait_s.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    deadline = time.time() + max_wait_s
    pending = list(export_reqs)

    while pending:
        if time.time() > deadline:
            raise TimeoutError(
                f"JSOC export timed out after {max_wait_s}s; "
                f"{len(pending)} requests still pending"
            )
        still_pending = []
        for item in pending:
            req = item["req"]
            req.wait(timeout=30)
            if req.status == "complete":
                log.info("Export %s complete (%d files) -- downloading",
                         req.id, len(req.urls))
                paths = req.download(str(out_dir), progress=True)
                for p in paths:
                    p = Path(p)
                    ts = _timestamp_from_fits(p)
                    results.append({
                        "channel": item["channel"],
                        "timestamp": ts,
                        "file_path": str(p),
                    })
            elif req.status in ("failed", "error", "canceled"):
                log.error("Export %s FAILED: %s", req.id, req.status)
            else:
                still_pending.append(item)
        pending = still_pending
        if pending:
            log.info("%d requests still pending -- waiting 60s", len(pending))
            time.sleep(60)

    log.info("Downloaded %d FITS files total", len(results))
    return results


def _timestamp_from_fits(p: Path) -> Optional[str]:
    """Read T_REC or DATE-OBS from FITS header (best-effort)."""
    try:
        import astropy.io.fits as fits
        with fits.open(str(p)) as hdul:
            for hdu in hdul:
                t = hdu.header.get("T_REC") or hdu.header.get("DATE-OBS")
                if t:
                    return str(t)
    except Exception:
        pass
    return None


def _validate_resolution(fits_records: list, target: int) -> None:
    """Spot-check that FITS files are target x target (D5 server-side guard)."""
    if not fits_records:
        return
    try:
        import astropy.io.fits as fits
        for rec in fits_records[:5]:
            p = Path(rec["file_path"])
            if not p.exists():
                continue
            with fits.open(str(p)) as hdul:
                # Data may be in extension 1 (image HDU) or extension 0
                data = None
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data
                        break
            if data is None:
                continue
            shape = data.shape[-2:]
            assert shape == (target, target), (
                f"Expected ({target},{target}), got {shape} in {p.name}. "
                "Check that JSOC im_patch rescale is working correctly."
            )
        log.info("Resolution check passed: sampled files are %dx%d", target, target)
    except ImportError:
        log.warning("astropy not installed; skipping resolution validation")


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_ar_identity_table(sharp_df: pd.DataFrame, noaa_map: dict,
                              noaa_events: pd.DataFrame,
                              cfg: dict) -> pd.DataFrame:
    """Build per-(HARPNUM, timestamp) table with label and position.

    Slides a sample window every sample_stride_hours along each AR's series.
    label_ge_M_within_24h = 1 iff a >=M1.0 flare from that AR's NOAA number
    starts within (t_sample, t_sample + predict_window_hours].
    """
    predict_h = cfg.get("predict_window_hours", 24)
    stride_h = cfg.get("sample_stride_hours", 1)
    stride = pd.Timedelta(hours=stride_h)

    rows = []
    for harpnum, grp in sharp_df.groupby("harpnum"):
        noaa_ar = noaa_map.get(int(harpnum), 0)
        times = pd.DatetimeIndex(sorted(grp["t_rec"].unique()))
        lon = grp["lon_fwt"].median()
        lat = grp["lat_fwt"].median()

        t = times.min()
        while t <= times.max():
            label = _label(t, noaa_ar, noaa_events, predict_h)
            rows.append({
                "harpnum": int(harpnum),
                "noaa_ar": noaa_ar,
                "timestamp": t,
                "lon_fwt": round(float(lon), 2),
                "lat_fwt": round(float(lat), 2),
                "label_ge_M_within_24h": label,
            })
            t += stride

    df = pd.DataFrame(rows)
    if not df.empty:
        n_pos = df["label_ge_M_within_24h"].sum()
        log.info(
            "AR identity table: %d samples, %d ARs, %d positive (%.1f%%)",
            len(df), df["harpnum"].nunique(), n_pos,
            100 * n_pos / len(df),
        )
    return df


def _label(t_sample, noaa_ar: int, noaa_events: pd.DataFrame,
            predict_h: int) -> int:
    """1 if any >=M1.0 flare from noaa_ar starts in (t_sample, t_sample+predict_h]."""
    if noaa_events.empty or noaa_ar == 0:
        return 0
    t_end = t_sample + pd.Timedelta(hours=predict_h)
    mask = (
        (noaa_events["noaa_ar"] == noaa_ar)
        & (noaa_events["start_time"] > t_sample)
        & (noaa_events["start_time"] <= t_end)
        & noaa_events["is_ge_M1"]
    )
    return int(mask.any())


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(identity_df: pd.DataFrame, fits_records: list,
                    out_dir: Path) -> Path:
    """Write manifest.csv mapping each (harpnum, timestamp, channel) to a FITS path.

    Schema: instance_id, harpnum, noaa_ar, timestamp, lon_fwt, lat_fwt,
            channel, file_path, label_ge_M_within_24h.

    Each full-disk FITS file appears once per AR active at that timestep,
    so downstream feature extraction can crop to the right AR.
    Timestamps are matched with a +-30 min tolerance.
    """
    if not fits_records:
        log.warning("No FITS records -- manifest will be empty")
        manifest_path = out_dir / "manifest.csv"
        pd.DataFrame(columns=[
            "instance_id", "harpnum", "noaa_ar", "timestamp",
            "lon_fwt", "lat_fwt", "channel", "file_path",
            "label_ge_M_within_24h",
        ]).to_csv(manifest_path, index=False)
        return manifest_path

    # Build (timestamp_str -> {channel -> file_path}) lookup from FITS records
    fits_lookup: dict = {}
    for rec in fits_records:
        ts_str = str(rec.get("timestamp", ""))
        ch = rec["channel"]
        fpath = rec["file_path"]
        fits_lookup.setdefault(ts_str, {})[ch] = fpath

    # Parsed timestamps for proximity matching
    parsed_fits_times: list = []
    for ts_str in fits_lookup:
        try:
            parsed_fits_times.append(pd.Timestamp(ts_str, tz="UTC"))
        except Exception:
            parsed_fits_times.append(None)
    fits_ts_keys = list(fits_lookup.keys())

    rows = []
    instance_id = 0
    tolerance = pd.Timedelta(minutes=30)

    for _, id_row in identity_df.iterrows():
        t = id_row["timestamp"]
        if not hasattr(t, "tz") or t.tzinfo is None:
            t = t.tz_localize("UTC")

        # Find closest FITS timestamp within tolerance
        best_key = None
        best_delta = tolerance
        for ts_str, ts_parsed in zip(fits_ts_keys, parsed_fits_times):
            if ts_parsed is None:
                continue
            delta = abs(ts_parsed - t)
            if delta < best_delta:
                best_delta = delta
                best_key = ts_str

        if best_key is None:
            continue

        for ch, fpath in fits_lookup[best_key].items():
            rows.append({
                "instance_id": instance_id,
                "harpnum": id_row["harpnum"],
                "noaa_ar": id_row["noaa_ar"],
                "timestamp": t.isoformat(),
                "lon_fwt": id_row.get("lon_fwt", float("nan")),
                "lat_fwt": id_row.get("lat_fwt", float("nan")),
                "channel": ch,
                "file_path": fpath,
                "label_ge_M_within_24h": id_row["label_ge_M_within_24h"],
            })
            instance_id += 1

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    log.info("Manifest: %s  (%d rows)", manifest_path, len(rows))
    return manifest_path


def validate_manifest(manifest_path: Path, expected_channels: list) -> None:
    """Assert:
    - Every file_path in the manifest exists on disk.
    - Every timestamp has entries for all expected channels (warn if not).
    """
    df = pd.read_csv(manifest_path)
    if df.empty:
        log.warning("Manifest is empty -- nothing to validate")
        return

    missing_files = [
        row["file_path"]
        for _, row in df.iterrows()
        if row["file_path"] and not Path(str(row["file_path"])).exists()
    ]
    if missing_files:
        raise AssertionError(
            f"Manifest references {len(missing_files)} missing files: "
            f"{missing_files[:3]} ..."
        )

    if expected_channels and "channel" in df.columns:
        ts_channels = df.groupby("timestamp")["channel"].apply(set)
        expected = set(expected_channels)
        incomplete = [
            ts for ts, chs in ts_channels.items() if not expected.issubset(chs)
        ]
        if incomplete:
            log.warning(
                "%d timestamps missing some channels (e.g. %s)",
                len(incomplete), incomplete[0],
            )

    log.info(
        "Manifest validation passed: %d rows, %d channels, %d ARs",
        len(df),
        df["channel"].nunique() if "channel" in df.columns else 0,
        df["harpnum"].nunique() if "harpnum" in df.columns else 0,
    )


# ---------------------------------------------------------------------------
# Mock mode -- offline dev/test, no JSOC required
# ---------------------------------------------------------------------------

def _run_mock(cfg: dict, window: dict, out_dir: Path) -> None:
    """Generate synthetic FITS stubs and identity table without JSOC.

    Mirrors the real fetch output schema so downstream code is testable locally.
    Always uses 1 channel and 1 day (fastest possible smoke-test).
    """
    import numpy as np

    try:
        import astropy.io.fits as fits
        _has_fits = True
    except ImportError:
        log.warning("astropy not available; writing empty .fits stubs")
        _has_fits = False

    resolution = cfg["resolution"]
    channels = _all_channels(cfg)[:1]   # 1 channel for speed
    start = pd.Timestamp(window["start"], tz="UTC")
    end = start + pd.Timedelta(days=1)  # 1 day only
    cadence = pd.Timedelta(minutes=cfg["cadence_minutes"])
    rng = np.random.default_rng(0)

    # Two synthetic HARPs with disjoint NOAA AR numbers
    mock_harps = [
        {"harpnum": 9001, "noaa_ar": 12192, "lon_fwt": 10.0, "lat_fwt": -15.0},
        {"harpnum": 9002, "noaa_ar": 12193, "lon_fwt": -20.0, "lat_fwt": 5.0},
    ]
    log.info("Mock: %d synthetic HARPs, 1 channel (%s), 1 day",
             len(mock_harps), channels[0])

    fits_records = []
    ch = channels[0]
    ch_dir = out_dir / "fits" / window["id"] / ch
    ch_dir.mkdir(parents=True, exist_ok=True)

    t = start
    while t <= end:
        fname = f"{ch}_{t.strftime('%Y%m%dT%H%M%S')}.fits"
        fpath = ch_dir / fname
        if _has_fits:
            data = rng.integers(0, 1000, (resolution, resolution), dtype="int16")
            hdu = fits.PrimaryHDU(data)
            hdu.header["T_REC"] = t.strftime("%Y.%m.%d_%H:%M:%S_TAI")
            hdu.header["CHANNEL"] = ch
            hdu.header["NAXIS1"] = resolution
            hdu.header["NAXIS2"] = resolution
            hdu.writeto(str(fpath), overwrite=True)
        else:
            fpath.write_bytes(b"SIMPLE  =                    T / mock fits stub")
        fits_records.append({
            "channel": ch,
            "timestamp": t.isoformat(),
            "file_path": str(fpath),
        })
        t += cadence

    # Synthetic identity table (random labels)
    id_rows = []
    for harp in mock_harps:
        t = start
        while t <= end:
            id_rows.append({
                "harpnum": harp["harpnum"],
                "noaa_ar": harp["noaa_ar"],
                "timestamp": t,
                "lon_fwt": harp["lon_fwt"],
                "lat_fwt": harp["lat_fwt"],
                "label_ge_M_within_24h": int(rng.random() < 0.15),
            })
            t += pd.Timedelta(hours=1)
    identity_df = pd.DataFrame(id_rows)

    identity_csv = out_dir / f"ar_identity_{window['id']}.csv"
    identity_df.assign(timestamp=identity_df["timestamp"].astype(str)).to_csv(
        identity_csv, index=False
    )

    manifest_path = write_manifest(identity_df, fits_records, out_dir)
    # validate_manifest skipped in mock mode (files exist but have synthetic content)

    _print_summary(manifest_path, identity_df, fits_records, cfg, window["id"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_jsoc_email() -> str:
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        raise RuntimeError(
            "JSOC_EMAIL environment variable not set. "
            "Register at https://jsoc.stanford.edu/ajax/register_email.html "
            "and set JSOC_EMAIL=<your-registered-email> before running. "
            "On Kaggle, store it in Kaggle Secrets and read it in the notebook."
        )
    return email


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _print_summary(manifest_path: Path, identity_df: pd.DataFrame,
                    fits_records: list, cfg: dict, window_id: str) -> None:
    n_files = len(fits_records)
    n_ts = len({r["timestamp"] for r in fits_records})
    n_ch = len({r["channel"] for r in fits_records})
    n_ars = identity_df["harpnum"].nunique() if not identity_df.empty else 0
    size_mb = sum(
        Path(r["file_path"]).stat().st_size
        for r in fits_records
        if Path(r["file_path"]).exists()
    ) / 1e6

    pos_rate = (
        identity_df["label_ge_M_within_24h"].mean()
        if not identity_df.empty else float("nan")
    )

    print()
    print("=" * 60)
    print("  Track A data fetch complete")
    print(f"  git SHA      : {_git_sha()}")
    print(f"  window       : {window_id}  "
          f"({cfg.get('_window_start', '?')} -> {cfg.get('_window_end', '?')})")
    print(f"  FITS files   : {n_files}  ({size_mb:.1f} MB)")
    print(f"  timesteps    : {n_ts}")
    print(f"  channels     : {n_ch}")
    print(f"  ARs          : {n_ars}")
    print(f"  base rate    : {pos_rate:.4f}")
    print(f"  manifest     : {manifest_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str, dry_run: bool = False, mock: bool = False) -> None:
    cfg = load_config(config_path)

    window_id = cfg["fetch_order"][0]
    window = next(w for w in cfg["event_windows"] if w["id"] == window_id)
    cfg["_window_start"] = window["start"]
    cfg["_window_end"] = window["end"]

    log.info("git SHA  : %s", _git_sha())
    log.info("Window   : %s  %s -> %s", window_id, window["start"], window["end"])
    log.info("Mode     : %s",
             "MOCK" if mock else ("DRY-RUN (1 day, 1 channel)" if dry_run else "FULL"))

    if mock:
        _run_mock(cfg, window, OUT_DIR)
        return

    # ------------------------------------------------------------------
    # Real fetch (requires JSOC_EMAIL + internet)
    # ------------------------------------------------------------------
    import drms

    jsoc_email = _get_jsoc_email()
    client = drms.Client(email=jsoc_email)

    # Step 1: SHARP metadata
    log.info("Step 1/5: SHARP metadata query")
    sharp_df = build_sharp_query(client, window, cfg)
    if sharp_df.empty:
        log.error("No qualifying ARs found for this window -- aborting")
        return
    noaa_map = build_noaa_map(sharp_df)
    log.info("Qualifying HARPs: %s", sorted(noaa_map.keys()))

    # Step 2: NOAA flare events (labels)
    log.info("Step 2/5: NOAA flare events")
    noaa_events = fetch_noaa_events(window["start"], window["end"])

    # Step 3: GOES X-ray flux (feature)
    log.info("Step 3/5: GOES X-ray flux")
    goes_flux = fetch_goes_flux(window["start"], window["end"])
    if not goes_flux.empty:
        goes_path = OUT_DIR / f"goes_flux_{window_id}.csv"
        goes_flux.to_csv(goes_path, index=False)
        log.info("GOES flux saved: %s  (%d rows)", goes_path, len(goes_flux))

    # Step 4: Image export (server-side rescale, D5)
    log.info("Step 4/5: JSOC image export")
    fetch_start = window["start"]
    fetch_end = (
        (pd.Timestamp(window["start"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if dry_run else window["end"]
    )
    channels_aia = cfg["channels_aia"][:1] if dry_run else cfg["channels_aia"]
    include_hmi = cfg.get("include_hmi_magnetogram", True) and not dry_run

    queries = _build_queries(
        fetch_start, fetch_end, channels_aia, include_hmi,
        cfg["cadence_minutes"],
    )
    export_reqs = request_image_export(
        client, queries, cfg["resolution"], jsoc_email
    )
    fits_dir = OUT_DIR / "fits" / window_id
    fits_records = poll_and_download(export_reqs, fits_dir)

    # D5 guard: assert server-side rescale produced correct resolution
    _validate_resolution(fits_records, cfg["resolution"])

    # Step 5: AR identity table + manifest
    log.info("Step 5/5: AR identity table + manifest")
    identity_df = build_ar_identity_table(sharp_df, noaa_map, noaa_events, cfg)
    identity_csv = OUT_DIR / f"ar_identity_{window_id}.csv"
    identity_df.assign(timestamp=identity_df["timestamp"].astype(str)).to_csv(
        identity_csv, index=False
    )
    log.info("AR identity saved: %s", identity_csv)

    all_ch = [f"AIA_{w}" for w in channels_aia]
    if include_hmi:
        all_ch.append("HMI_M")
    manifest_path = write_manifest(identity_df, fits_records, OUT_DIR)
    validate_manifest(manifest_path, all_ch)

    _print_summary(manifest_path, identity_df, fits_records, cfg, window_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Track A / A1 -- JSOC event-window data acquisition"
    )
    ap.add_argument("--config", required=True, help="Path to data_v1.yaml")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Fetch 1 day + 1 channel only (validates the export/rescale path)",
    )
    ap.add_argument(
        "--mock", action="store_true",
        help="Generate synthetic data locally without JSOC (offline dev/test)",
    )
    args = ap.parse_args()
    main(args.config, dry_run=args.dry_run, mock=args.mock)
