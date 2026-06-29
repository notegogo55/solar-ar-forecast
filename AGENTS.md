# AGENTS.md — operating brief for the Antigravity agent

You are working on **Automated Solar Active Region Tracking & Eruption Forecasting** (deep learning, multi-wavelength SDO observations). Read this file first, then `docs/workplan.md` (the locked decisions D1–D9) and `docs/runbook-kaggle.md` (the compute workflow) before writing code.

## Prime directives
1. **Respect the locked decisions below.** They were settled deliberately. If a task seems to require breaking one, stop and flag it — do not silently override.
2. **Never put data or weights in git.** No `.fits`, `.pth`, `.h5`, checkpoints, or datasets. They live in Kaggle Datasets. `.gitignore` enforces this; don't weaken it.
3. **Never run heavy training locally.** The local GPU is a GTX 1650 (4 GB) — dev + Track B only. All segmentation training (U-Net/Surya/SAM2) runs on Kaggle (16 GB). `src.config.assert_can_train_seg()` guards this.
4. **Antigravity's sandbox has no GPU.** Don't attempt GPU jobs in the agent sandbox; GPU work goes to Kaggle via the `kaggle/` launchers (see runbook secs 5 & 8).
5. **Don't break the validation protocol (D8).** It's the project's credibility. See `src/eval/`.

## Locked decisions (full detail in docs/workplan.md)
- **D1** "Done" = a working end-to-end pipeline; accuracy is secondary; breadth is stretch.
- **D2/D3** Segmentation is on the critical path; all three models (U-Net / Surya / SAM2) run, sharing ONE label set + ONE held-out split. IoU/Dice do NOT gate the spine — first model to clear "good enough" unblocks tracking. U-Net is the cheap unblocker; Surya/SAM2 are Kaggle-only comparison runs.
- **D4** Pixel labels from Surya's packaged AR-seg dataset; AR identity from SHARP HARPNUM + NOAA AR numbers. Two sources, two jobs.
- **D5** Full-disk 1024² data, event-windows only, 1 h cadence, server-side rescale. Known limitation: max-intensity dilution from downsampling (stretch fix = native cutouts).
- **D6** Topology: Antigravity (local, dev) + local 1650 (Track B) + Kaggle (Track A GPU + data/checkpoints). Chalawan/ADA is an upgrade if it opens up.
- **D7** v1 forecast target = binary "≥M1.0 flare within next 24 h" per tracked AR. NOAA flare-AR list = label; GOES flux = feature. Other configs (72 h, X-class, CME) are stretch.
- **D8** Strict validation: time-block + whole-AR split (no AR straddles folds); resample/weight TRAIN only; metrics TSS/HSS/BSS + reliability (never headline accuracy); one held-out window evaluated once.
- **D9** Forecaster = Holt-Winters + LSTM ensemble on scalar per-AR feature time-series. Develop & validate on SWAN-SF first, then swap in real features.

## Two-track structure (build in parallel)
- **Track A (image pipeline):** `src/data` → `src/seg` → `src/track` → `src/features`. The long pole; tracking (`src/track`, A3) is the #1 schedule risk.
- **Track B (forecasting):** `src/forecast` + `src/eval`. Runs locally on SWAN-SF, de-risks independently. Start this early.
- **Integration:** features from Track A → validated Track-B forecaster → end-to-end ≥M/24 h. That's the G4 "done" milestone.

## Conventions
- All paths/device come from `src/config.py` — import it, never hard-code.
- Experiments are yaml-driven (`configs/`). Add a new yaml rather than hard-coding hyperparameters.
- Kaggle kernels stay thin (`kaggle/run_*.py`): clone repo → pip install → `python -m src...`. Logic lives in `src/`.
- Every training loop checkpoints to `OUT_DIR` each epoch and resumes from `CKPT_IN` (Kaggle sessions cap ~12 h).
- Log the git commit SHA in every Kaggle run.

## Suggested first tasks
1. Implement `src/eval/splits.py` + `src/eval/metrics.py` (the validation harness) — needed by Track B immediately.
2. Implement `src/forecast/train.py` against SWAN-SF and validate it. This is the lowest-risk, highest-leverage starting point.
3. In parallel, draft `src/data/fetch.py` (drms export) and test it on the single 2014 event-window in `configs/data_v1.yaml`.
