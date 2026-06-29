# Kickoff prompt 05 — feature extraction + integration → G4 spine (Track A / A4 + INT)

Scope: extract per-AR features and plug them into the validated Track-B forecaster — this **closes the end-to-end spine = G4 "done"**. Feature extraction + integration are light → mostly local. Conventions are set by now, so this can run mostly-autonomous with a checkpoint at the plan and before the final evaluation.
Paste the block below into the Antigravity agent.

---

You are working in the `solar-ar-forecast` repo. Before doing anything, read `AGENTS.md`, then `docs/workplan.md` (decisions D1–D9; this milestone is the G4 definition of done) and `docs/runbook-kaggle.md`. Follow every rule in `AGENTS.md`.

WORKING STYLE.
Produce a brief implementation plan + file list first (quick approval). Then implement mostly autonomously, but PAUSE for my review before the final held-out evaluation run.

TASK (this milestone only):
1. `src/features/` — project each tracked HMI AR mask onto the co-temporal AIA channels, then extract a per-AR feature time-series: **maximum** pixel intensity per channel (use max, not mean — D5), total magnetic flux, and area, over the event-window. Label each sequence with "≥M1.0 flare within next 24 h" via the NOAA flare-AR list (D7).
2. INTEGRATION — feed these real features into the validated Track-B forecaster (the Holt-Winters + LSTM ensemble from prompt 01), reusing `src/eval` unchanged. Run the full pipeline end to end: data → tracked ARs → features → forecast.
3. Evaluate on a time-blocked held-out window with the strict protocol (D8): whole-AR split, resample TRAIN only, report TSS/HSS/BSS + reliability at the true base rate, evaluated once.

VERIFY.
- Assert the feature is the per-AR MAX (not mean) and that feature time-series shapes/timestamps align with the AR identity table.
- The end-to-end run prints TSS/HSS/BSS + a reliability table on the held-out window at the true (unbalanced) base rate.
- Compare against the SWAN-SF benchmark result from prompt 01 (sanity, not a hard gate).

CONVENTIONS. Paths/device from `src/config.py`. Reuse `src/eval` and the Track-B forecaster as-is — do not fork the validation harness.

OUT OF SCOPE (do NOT touch). Stretch only — no 72 h horizon, no X-class, no CME classification, no 6-case AIA ablation, no native-resolution cutout features. Never stage data/weights for git commit.

DEFINITION OF DONE (this is G4). End-to-end pipeline runs from 1024² data through tracked ARs → per-AR feature time-series → ≥M/24 h forecast → TSS/HSS/BSS + reliability at the true base rate on a held-out window evaluated once; reuses `src/eval` unchanged; nothing out-of-scope modified; nothing staged to git. Report final metrics, the data→forecast path, and files changed.
