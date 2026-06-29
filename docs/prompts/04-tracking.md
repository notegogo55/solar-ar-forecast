# Kickoff prompt 04 — instance separation + temporal tracking (Track A / A3)

Scope: turn semantic masks into per-AR instances with persistent identity across time. **This is the #1 schedule risk in the workplan.** Compute is light (CV/geometry, no GPU) → develop **locally** on a pulled subset of seg masks. Plan-first, with explicit failure-mode analysis.
Paste the block below into the Antigravity agent.

---

You are working in the `solar-ar-forecast` repo. Before doing anything, read `AGENTS.md`, then `docs/workplan.md` (decisions D1–D9, note A3 is flagged as the highest schedule risk) and `docs/runbook-kaggle.md`. Follow every rule in `AGENTS.md`.

WORKING STYLE — PLAN FIRST, AND DE-RISK EXPLICITLY.
Do not write code yet. First produce an implementation plan AND a short list of the failure modes you expect (identity swaps, merges/splits of ARs, fast emergence, limb foreshortening) and how each test will catch them. Wait for my approval before coding.

TASK (this milestone only) — implement `src/track/`, per decision D4:
1. Instance separation: split each semantic AR mask into individual ARs (connected components + morphological cleanup; handle touching/merging regions).
2. Temporal tracking: maintain persistent per-AR identity across the time series using temporal IoU + differential-rotation compensation, ANCHORED to SHARP HARPNUM / NOAA AR numbers (the only identity source that survives rotation).
3. Output: per-AR tracked patches with a stable identity column over the whole event-window.

VALIDATE HARD (this is the risky one).
- Unit test on a synthetic 2-AR moving sequence: identities are preserved, no swap.
- On real data: report an agreement metric between your tracked identities and the HARPNUM/NOAA ground-truth association; flag every disagreement for inspection.
- Assert no identity swaps across the window for at least one known multi-AR case.

CONVENTIONS. Paths/device from `src/config.py` — this runs on CPU, develop locally on a small subset of seg masks pulled from Kaggle. Config-driven. Log assumptions about rotation model + thresholds.

OUT OF SCOPE (do NOT touch). No re-training segmentation, no feature extraction, no forecasting. Never stage data for git commit.

DEFINITION OF DONE. Plan + failure-mode list approved; instance separation + tracking produce stable per-AR identities; validated against HARPNUM/NOAA with a reported agreement metric; synthetic + real-case tests pass; remaining failure modes documented; nothing out-of-scope modified; nothing staged to git.
