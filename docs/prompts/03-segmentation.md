# Kickoff prompt 03 — segmentation ×3 (Track A / A2)

Scope: train/compare the three segmentation models on a **shared label set + shared held-out split**. Runs on a **Kaggle GPU kernel** (run_train.py launcher) — NOT on the local 1650. Plan-first.
Paste the block below into the Antigravity agent.

---

You are working in the `solar-ar-forecast` repo. Before doing anything, read `AGENTS.md`, then `docs/workplan.md` (decisions D1–D9) and `docs/runbook-kaggle.md`. Follow every rule in `AGENTS.md`.

WORKING STYLE — PLAN FIRST.
Do not write code yet. First produce an implementation plan: files, the shared-label/shared-split design, the checkpoint/resume scheme, and the smoke test. Wait for my approval before coding.

TASK (this milestone only) — implement `src/seg/train.py` supporting `model ∈ {unet, surya, sam2}`, per decisions D3/D4:
1. One SHARED label set (Surya's packaged AR-seg labels) and one IDENTICAL held-out split used by all three — otherwise IoU/Dice aren't comparable.
2. U-Net = the cheap "unblock the spine" model (trains fast). Surya = LoRA + gradient checkpointing so it fits Kaggle's 16 GB. SAM2 = zero-shot inference, no training.
3. Score IoU/Dice on the shared held-out split for each model and write a comparison table. **IoU/Dice do NOT gate the spine** — the first model clearing a stated "good enough" bar unblocks tracking; the comparison is a reported result, not a blocker.
4. Must call `src.config.assert_can_train_seg()` (refuses to run on the 1650). Checkpoint to `OUT_DIR` every epoch and resume from `CKPT_IN` — Kaggle sessions cap ~12 h (runbook sec 6). Read data from the attached `solar-event-windows-v1` Dataset.

VERIFY EARLY. Before any long run, confirm **Surya's expected input resolution/format** — if it requires native res, run the seg subsystem on Surya's own data spec rather than the 1024² forecasting data (A2 risk in the workplan). Add a smoke test: one training step + a checkpoint save→resume round-trip works.

CONVENTIONS. Paths/device from `src/config.py`. Config-driven via `configs/unet.yaml` / `configs/surya.yaml` (+ a sam2 config). Log the git commit SHA in every run.

OUT OF SCOPE (do NOT touch). No instance separation/tracking, no feature extraction, no forecasting. Never run seg training locally. Never stage weights/data for git commit.

DEFINITION OF DONE. Plan approved; all three models produce masks scored on the IDENTICAL held-out split with a comparison table; checkpoint/resume verified; at least one model clears the stated "good enough" IoU bar to unblock tracking; results + git SHA reported; nothing out-of-scope modified; nothing staged to git.
