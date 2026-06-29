# Kickoff prompt 01 — eval harness + Track B forecaster (SWAN-SF)

Scope: first milestone. Local-only (no Kaggle/GPU/data-download). Plan-first. Test-first for the eval harness.
Paste the block below into the Antigravity agent.

---

You are working in the `solar-ar-forecast` repo. Before doing anything, read `AGENTS.md`, then `docs/workplan.md` (decisions D1–D9) and `docs/runbook-kaggle.md`. Follow every rule in `AGENTS.md`.

WORKING STYLE — PLAN FIRST.
Do not write code yet. First produce an implementation plan: the files you will create/modify, the public functions/classes in each, the test cases you will write, and any libraries to add to requirements.txt. Wait for my approval before coding.

TASK (this milestone only) — local, image-free forecasting track (Track B) on SWAN-SF [the Angryk et al. 2020 SWAN-SF dataset of SHARP-parameter time series]:
1. SWAN-SF loading into per-AR multivariate time series (cache under ./data, gitignored).
2. src/eval/splits.py — time_block_whole_ar_split(...): chronological time-block split where every AR's full set of windows lands in exactly one fold; split points must not cut through an AR's disk passage.
3. src/eval/metrics.py — tss, hss, bss, and a reliability-diagram helper.
4. src/forecast/train.py — Holt-Winters baseline + LSTM, combined as an ensemble (D9); target = "ge M1.0 flare within the next 24 h" (D7). Resample/weight TRAIN ONLY; the held-out set keeps the true base rate (D8).
5. A run that trains on SWAN-SF and prints TSS / HSS / BSS + a reliability table on ONE held-out window, evaluated once.

TEST-FIRST FOR THE EVAL HARNESS. Before implementing metrics.py and splits.py, write tests:
- metrics: tss/hss/bss match hand-computed values for >=2 small confusion-matrix / probability examples.
- splits: no AR id appears in both train and test on a synthetic dataset.
- a guard (test or runtime assertion) that resampling did not change the test-set base rate.
All tests must pass before you call the task done.

CONVENTIONS. Read all paths/device from src/config.py (never hard-code). Make the run config-driven via configs/lstm_swansf.yaml. Keep it runnable on CPU / 4 GB GPU — this track is local by design.

OUT OF SCOPE (do NOT touch). No Track A: no src/data/fetch.py / drms, no src/seg segmentation, no tracking, no feature extraction, no Kaggle kernels. Never stage data, FITS, or checkpoints for commit.

DEFINITION OF DONE. Plan approved; eval-harness tests written first and passing; forecast/train.py runs on real SWAN-SF and reports TSS/HSS/BSS + reliability on a held-out window at the true base rate; everything reads from config.py + configs/lstm_swansf.yaml; nothing out-of-scope modified; nothing data/weights staged. Report final metrics + files changed.

---

## Reusing this pattern for later milestones
Keep the same skeleton — read AGENTS.md first / plan-first / scoped task / out-of-scope list / acceptance criteria — and swap the TASK block. Next likely prompts: `02` data fetch (src/data/fetch.py on the 2014 window), `03` segmentation on Kaggle (src/seg), `04` tracking (src/track, the high-risk one).
