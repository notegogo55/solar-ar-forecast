# Kickoff prompt 02 — data acquisition (Track A / A1)

Scope: fetch the forecasting dataset. Runs on a **Kaggle dataprep kernel** (internet ON, GPU off) — NOT from the home connection (runbook sec 3). Plan-first.
Paste the block below into the Antigravity agent.

---

You are working in the `solar-ar-forecast` repo. Before doing anything, read `AGENTS.md`, then `docs/workplan.md` (decisions D1–D9) and `docs/runbook-kaggle.md`. Follow every rule in `AGENTS.md`.

WORKING STYLE — PLAN FIRST.
Do not write code yet. First produce an implementation plan: files to create/modify, public functions, the dry-run you will use to validate before the full fetch, and any libraries to add. Wait for my approval before coding.

TASK (this milestone only) — implement `src/data/fetch.py`, the full-disk event-window acquisition for Track A, per decision D5:
1. Via `drms`/JSOC, export **full-disk 1024²** images with **server-side rescale** (no local downsampling), HMI line-of-sight magnetogram + the 8 AIA channels in `configs/data_v1.yaml`, at **1 h cadence**, for the window(s) in `fetch_order` — start with `w1_oct2014` (2014-10-17 → 2014-10-30, AR 12192's disk passage). Handle the async export correctly: request → poll → download.
2. Ingest the label/feature sources separately (D4/D7): GOES X-ray flux (feature), the NOAA/SWPC flare event list with NOAA AR association (label), and SHARP HARPNUM ↔ NOAA AR identity.
3. Write a `manifest.csv` mapping every (timestamp, channel) to its file path + the per-timestamp AR identity table.
4. Designed to run as the Kaggle dataprep kernel: read the JSOC email from a Kaggle Secret; write outputs to `OUT_DIR` (=/kaggle/working) so a "Save Version" produces the Dataset named `out_dataset` (solar-event-windows-v1).

SAMPLING & FILTERS (read them from `configs/data_v1.yaml`, do not hard-code).
- Sampling philosophy: take ALL qualifying HARPs in the window; keep the natural class imbalance; never hand-balance ARs (resampling happens TRAIN-only, later, in Track B).
- Per-AR / per-timestep filters: sample only timesteps within ±70° of central meridian (`central_meridian_deg`); require a NOAA AR number (`require_noaa_number`); keep an AR only if it has ≥48 h of valid in-window, in-longitude frames (`min_coverage_hours`); drop frames flagged by the SHARP `QUALITY` keyword.
- Labeling: a sample is positive iff a ≥M1.0 flare occurs within the next 24 h for that AR (`predict_window_hours`); lookback `lookback_hours`; emit a sliding sample every `sample_stride_hours`.

VALIDATE BEFORE THE FULL FETCH. Start with a tiny dry-run (1 day, 1 channel) that confirms the export path, rescale, and manifest schema. Add: an assertion that all channels for a timestamp are co-aligned and the same shape; a check that the manifest references every requested (timestamp, channel).

CONVENTIONS. All paths/device from `src/config.py` (never hard-code). Config-driven via `configs/data_v1.yaml`. Internet must be ON for this kernel. Log the git commit SHA.

OUT OF SCOPE (do NOT touch). No segmentation, tracking, feature extraction, or forecasting. No full-disk download to the local machine. Never stage FITS/data for git commit.

DEFINITION OF DONE. Plan approved; dry-run validated; `fetch.py` runs the 2014 window on a Kaggle dataprep kernel; `manifest.csv` + AR identity table complete and consistent; output saved as the `solar-event-windows-v1` Dataset; nothing out-of-scope modified; nothing staged to git. Report dataset size, # timesteps/channels/ARs, and files changed.
