# Work Plan (Revised) — Automated Solar Active Region Tracking & Eruption Forecasting

**Project:** Automated Solar Active Region Tracking and Eruption Forecasting Using Multi-Wavelength Observations: A Deep Learning Approach
**Team:** NARIT Research Group (ILRS) — single-researcher execution
**Horizon:** ~26 weeks (6 months)
**Revision basis:** Restructured from the proposal's linear 7-phase plan into a **two-track, thin-slice-first** plan after a full design review. The driving constraint is *one researcher in 26 weeks*, so the plan protects a working end-to-end spine and pushes breadth to an explicit stretch backlog.

---

## 1. Locked decisions (design log)

These were resolved during review and drive everything below.

| # | Decision | Resolution |
|---|----------|------------|
| D1 | Definition of "done" | **Working end-to-end pipeline** is the deliverable; accuracy is secondary. Breadth = stretch. |
| D2 | Segmentation placement | On the **critical path** (researcher's call), not decoupled. |
| D3 | Which segmentation model | **All three in parallel** — Surya `ar_segmentation` downstream example, U-Net (baseline), SAM2 (zero-shot). Conditions: one shared label set + identical held-out split; asymmetric effort; **IoU/Dice does not gate the spine** — first model to clear "good enough" unblocks it. **On the Kaggle topology (D6): U-Net is the cheap "unblock-the-spine" model (fast, low quota); Surya/SAM2 are the expensive comparison runs — budget them against the ~30 GPU-hr/wk Kaggle quota and checkpoint for resumable training.** |
| D4 | Labels & identity | **Two sources, two jobs:** pixel labels from Surya's packaged AR-seg dataset; AR **identity** from SHARP HARPNUM + NOAA AR numbers. Manual annotation only for a small QA subset. |
| D5 | Data format | **Full-disk 1024²**, whole pipeline, v1. **Event-windows only** (not continuous), cadence **1 h**, server-side rescale. Known limitation logged: max-intensity dilution from downsampling. Stretch: native-res cutout features. |
| D6 | Dev environment & compute | **Primary topology (free):** Develop in **Google Antigravity** on the local laptop (NVIDIA GTX 1650, 4 GB — dev only; Intel UHD iGPU is unusable for DL). **Track B (LSTM/Holt-Winters) runs locally** on the 1650/CPU. **Track A GPU training runs on Kaggle** (free P100 16 GB / T4×2, ~30 GPU-hr/wk, persistent ~100 GB datasets). Antigravity's sandbox has **no GPU** → training scripts run in the host env / on Kaggle, never in the agent sandbox. The 1650 4 GB **cannot run Surya** (366 M) — that's why Surya/SAM2 live on Kaggle's 16 GB. Phase-0 gates: Kaggle account + GPU verified, GitHub repo for code sync, Kaggle Dataset for data staging, JSOC registration. **Upgrade path if it becomes available:** Chalawan/ADA via SSH (then 3-model-parallel becomes comfortable). |
| D7 | Forecast target (v1) | **Binary: "will this tracked AR produce a ≥M1.0 flare within the next 24 h?"** NOAA flare–AR list = label; GOES X-ray flux = input feature. Expansion configs deferred. |
| D8 | Validation protocol | **Strict, locked before modeling:** time-block + whole-AR split (no AR straddles folds); resample/weight **train only**, test keeps true base rate; metrics **TSS / HSS / BSS + reliability** (never headline accuracy); one final held-out window evaluated once; training curves only for convergence checks. |
| D9 | Forecasting model | **Scalar per-AR feature time-series** → Holt-Winters + LSTM **ensemble**. Develop & validate **first on SWAN-SF** (ready-made, no images), then swap in real extracted features. |

---

## 2. Structural insight — two parallel tracks meeting late

The single most important change vs. the proposal: the forecasting half does **not** have to wait for the image half.

- **Track A — Image / CV pipeline (the long pole):** data acquisition → segmentation (×3) → instance separation + temporal tracking → mask projection to AIA → per-AR feature extraction.
- **Track B — Forecasting (de-risks independently):** SWAN-SF → Holt-Winters + LSTM ensemble + the strict validation harness → a *validated forecaster waiting for features*.
- **Integration:** plug Track A's features into Track B's validated forecaster → end-to-end ≥M/24 h spine → evaluate.

Why this matters: even if Track A underperforms or slips, Track B still produces a validated model and a benchmark result. The spine is demonstrable at G4 regardless.

```
Wk:    1   3   5   7   9   11  13  15  17  19  21  23  25
P0  [==]                                                      setup + gates
B   [============================]                            SWAN-SF forecaster (Track B)
A1     [=========]                                            data acquisition
A2         [=============]                                    segmentation ×3
A3              [=============]                               instance + tracking (RISK)
A4                    [=============]                         projection + features
INT                              [=========]                  integration → thin-slice spine
P5                                     [=========]            hardening + stretch expansions
P6                                            [=======]       docs + final report
```

---

## 2a. Tooling & compute topology

| Layer | Tool | Role |
|-------|------|------|
| **Development / orchestration** | **Google Antigravity** on local laptop (GTX 1650 4 GB) | Agents scaffold the SunPy/drms pipeline, validation harness, and tests; Manager view runs Track A & Track B builds in parallel. Dev only — not a training GPU. |
| **Track B compute** | **Local 1650 / CPU** | LSTM + Holt-Winters on tabular SWAN-SF + feature tables run fine locally — zero external dependency. |
| **Track A GPU compute** | **Kaggle (free)** — P100 16 GB / T4×2 | U-Net / SAM2 / Surya LoRA training + segmentation inference. ~30 GPU-hr/wk quota, ~12 h/session. |
| **Data staging** | **Kaggle Datasets** (persistent ~100 GB) | Event-window FITS + extracted feature tables + checkpoints live here, so Kaggle notebooks read them without re-upload. |
| **Code sync** | **GitHub** | Antigravity commits locally → Kaggle notebook pulls the repo. |

**Working loop:** Antigravity agents edit code locally on the 1650 → commit to GitHub → a Kaggle notebook pulls the repo + reads data from a Kaggle Dataset → trains on P100/T4 → writes checkpoints/features back to Kaggle output → pull results down for the agent to verify. Track B stays entirely local.

**Discipline this topology demands:** (1) budget the ~30 GPU-hr/wk — do all dev/iteration locally or on CPU, spend Kaggle GPU only on real training; Surya runs are the expensive ones. (2) Kaggle sessions cap at ~12 h and can disconnect → **checkpoint to Kaggle output every epoch, resumable training mandatory.** (3) Keep v1 data small (1024² + few event-windows) so it fits the persistent dataset and trains within quota.

**Upgrade path:** if Chalawan/ADA access opens up, point training at it over SSH and the 3-model comparison becomes comfortable — no other plan changes.

---

## 3. Phase plan

### Phase 0 — Setup & de-risk gates · Weeks 1–2
- Install Antigravity locally; create Kaggle account + **verify a GPU notebook runs** (P100/T4); set up GitHub repo for code sync.
- Create the Kaggle Dataset that will hold staged data + checkpoints; confirm the local→GitHub→Kaggle loop works on a toy job.
- Register JSOC email for `drms` export.
- Local environment (SunPy, drms, PyTorch); clone Surya repo; experiment tracking.
- Finalize event-windows, the AR sample, and the **≥M1.0 / 24 h** label spec in writing.
- Reproduce a trivial reference forecast (climatology / persistence) **locally on the 1650** to anchor skill scores.

### Track B — Forecasting on SWAN-SF · Weeks 2–8 *(runs in parallel from Wk 2)*
- Wk 2–4: Acquire SWAN-SF; build Holt-Winters baseline + LSTM; implement the **strict validation harness** (D8).
- Wk 4–6: Tune the ensemble; handle class imbalance (train-only); lock the protocol; record a benchmark result.
- Wk 6–8: Freeze as the validated forecaster awaiting real features (slack/buffer here).

### Track A — Image pipeline · Weeks 3–16
- **A1 · Wk 3–6 — Data acquisition:** full-disk 1024² event-windows via drms/JSOC (server-side rescale), HMI magnetogram + 8 AIA channels @ 1 h; co-alignment, exposure normalization, quality flags; ingest SHARP/NOAA + flare–AR matching.
- **A2 · Wk 5–10 — Segmentation (×3):** run Surya AR-seg downstream example (primary); train U-Net baseline on the shared labels; SAM2 zero-shot; score IoU/Dice on the shared held-out split. *First to clear "good enough" unblocks the spine.* **Verify early that Surya's expected input resolution/format matches your pipeline; if it demands native res, run the seg subsystem on Surya's own data spec.**
- **A3 · Wk 8–13 — Instance separation + temporal tracking (highest schedule risk):** split semantic masks into per-AR instances; assign persistent identity via temporal IoU + differential-rotation compensation, anchored to HARPNUM/NOAA. Extra slack allocated here.
- **A4 · Wk 11–16 — Projection + feature extraction:** project HMI masks onto co-temporal AIA channels; extract per-AR max-intensity per channel + magnetic-flux + area over time → feature time-series; label sequences via NOAA flare–AR list.

### Integration — thin-slice spine closes · Weeks 15–19
- Plug extracted features into the validated Track-B forecaster.
- Run the end-to-end ≥M/24 h pipeline; evaluate on the held-out window with the strict protocol.
- **G4 — "Done" milestone (~Wk 19): working end-to-end pipeline with real TSS/HSS/BSS.**

### Phase 5 — Hardening + stretch expansions · Weeks 19–23
Work the stretch backlog (Section 6) in priority order, taking as many as time allows. Per-model visualizations + reliability diagrams.

### Phase 6 — Documentation & wrap-up · Weeks 24–26
Technical documentation, reproducible README, final report; optional dashboard.

---

## 4. Milestones & gates

| Gate | Week | Criterion | If missed |
|------|------|-----------|-----------|
| G0 | 2 | Kaggle GPU verified + data-staging loop (local→GitHub→Kaggle) working; JSOC registered | Fix the sync loop before any Track A training; Track B proceeds locally regardless |
| G1 | 8 | SWAN-SF forecaster validated (Track B de-risked) | Forecasting is independent; slip here does not block Track A |
| G2 | 10 | ≥1 segmentation model "good enough" | Fall back to whichever model is best so far; SHARP boxes as emergency mask |
| G3 | 13 | Tracking yields persistent AR identities | The #1 risk — escalate; lean harder on HARPNUM as identity |
| **G4** | **19** | **End-to-end ≥M/24 h spine with real skill scores** | **This is "done." Protect it above all stretch.** |
| G5 | 26 | Documentation + final report complete | — |

---

## 5. Revised WBS (mapped to proposal numbering)

| WBS | Task | Track / Phase |
|-----|------|---------------|
| 11000 | Environment, repo, experiment tracking | P0 |
| 12000 | Kaggle GPU + data-staging loop + JSOC registration | P0 (gate G0) |
| 13000 | Event-window & AR sample selection; label spec | P0 |
| 54000 | SWAN-SF forecaster + validation harness | **Track B** (moved early) |
| 51000 | Holt-Winters baseline | Track B |
| 52000 | LSTM model | Track B |
| 53000 | Ensemble + imbalance handling (train-only) | Track B |
| 21000 | SDO full-disk 1024² retrieval (event-windows) | A1 |
| 22000 | GOES / SHARP / NOAA ingestion + flare–AR matching | A1 |
| 23000 | Co-alignment + normalization | A1 |
| 31000 | Shared label set (Surya pixels) + QA subset | A2 |
| 32000 | U-Net baseline | A2 |
| 33000 | Surya AR-seg (primary) + SAM2 zero-shot | A2 |
| 34000 | Instance separation + temporal tracking (HARPNUM/NOAA) | A3 |
| 41000 | Mask projection to AIA | A4 |
| 42000 | Per-AR feature extraction | A4 |
| 43000 | Sequence labeling (NOAA list) | A4 |
| 61000 | Pipeline integration (features → forecaster) | INT |
| 62000 | Evaluation (TSS/HSS/BSS, reliability), held-out once | INT (gate G4) |
| 63000 | AIA channel ablation (6-case) | **Stretch** |
| 64000 | Per-model visualizations | P5 |
| 71000 | Technical documentation | P6 |
| 72000 | Final report | P6 |
| 73000 | Optional dashboard | Stretch |

---

## 6. Stretch backlog (post-G4, prioritized)

1. **72 h lead time** (second horizon config).
2. **X-class threshold** (the headline "extreme" target the project builds toward).
3. **6-case AIA layer ablation matrix** — now scientifically meaningful because the spine works.
4. **Native-resolution cutout features** — fixes the max-intensity dilution limitation from D5.
5. **CME classification** (DONKI / SOHO-LASCO CDAW) — confined vs. eruptive.
6. **Full Surya-vs-U-Net-vs-SAM2 comparison writeup** + SAM2 temporal propagation.
7. **Visualization dashboard.**

---

## 7. Risk register (top items)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Temporal tracking (A3) overruns | High | Anchor identity to HARPNUM/NOAA; extra slack; SHARP boxes as emergency fallback mask |
| Kaggle ~30 GPU-hr/wk quota throttles training | Med-High | Dev/iterate locally or on CPU; spend GPU only on real training; U-Net (cheap) unblocks the spine, Surya runs scheduled deliberately |
| Kaggle ~12 h session cap / disconnects | Medium | Checkpoint to Kaggle output every epoch; resumable training mandatory |
| Surya (366 M) won't fit local 1650 4 GB | Certain (design) | Surya/SAM2 train on Kaggle 16 GB, never locally; the 1650 is dev + Track B only |
| Antigravity sandbox has no GPU | Certain (design) | Training runs in the host env / on Kaggle by design — never inside the agent sandbox |
| Surya input-res mismatch | Medium | Verify in A2 Wk 5; run seg subsystem on Surya's own data spec if needed |
| max-intensity dilution from 1024² | Medium | Logged as known limitation; native-cutout stretch (#4) |
| Class imbalance produces illusory scores | Medium | Strict protocol (D8) already enforced; TSS-first reporting |
| Scope creep | Medium | Everything past G4 is explicitly stretch; G4 is protected |

---

## 8. Definition of Done (G4)

A single configuration-driven Python pipeline that, end to end and on a time-blocked held-out window: ingests full-disk 1024² SDO data for an event-window → segments and tracks active regions with persistent identity → extracts per-AR multi-wavelength feature time-series → forecasts **P(≥M1.0 flare within 24 h)** with an HW+LSTM ensemble → reports **TSS / HSS / BSS + a reliability diagram** at the true base rate. Everything beyond this is stretch.
