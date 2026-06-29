# Solar AR Tracking & Eruption Forecasting

Automated active-region tracking and ≥M-class flare forecasting from multi-wavelength SDO observations, using a deep-learning segmentation → tracking → feature → forecasting pipeline.

## Quick start
```bash
git clone https://github.com/<you>/solar-ar-forecast.git
cd solar-ar-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add torch locally if needed
cp .env.example .env                      # fill in JSOC_EMAIL
```

- **Track B (forecasting) runs locally:** `python -m src.forecast.train --config configs/lstm_swansf.yaml`
- **Track A (segmentation) runs on Kaggle:** see `docs/runbook-kaggle.md`.

## Where things are
| Path | What |
|------|------|
| `AGENTS.md` | Operating brief — read this first (esp. for the Antigravity agent) |
| `docs/workplan.md` | Full workplan: decisions D1–D9, phases, milestones, risks |
| `docs/runbook-kaggle.md` | local → GitHub → Kaggle workflow, step by step |
| `src/config.py` | Paths + device (auto-detects local vs Kaggle) |
| `configs/` | yaml per experiment |
| `kaggle/` | thin script-kernel launchers + `kernel-metadata.json` |

## Compute topology
Dev in **Antigravity** on the local laptop (GTX 1650 — dev + Track B only). GPU training for Track A runs on **Kaggle** (16 GB; Surya/SAM2/U-Net). Code syncs local → GitHub → Kaggle; data and checkpoints live in Kaggle Datasets and never enter git. See `AGENTS.md` for the rules.

## Definition of done (G4)
A config-driven pipeline that, on a time-blocked held-out window, goes from 1024² SDO data → tracked ARs → per-AR feature time-series → P(≥M1.0 within 24 h) via an HW+LSTM ensemble → TSS/HSS/BSS + reliability at the true base rate.
