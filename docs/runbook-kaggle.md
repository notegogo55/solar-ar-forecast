# Runbook — local → GitHub → Kaggle workflow

Companion to the project workplan. This is the operating manual for the free compute topology: **local laptop (GTX 1650) for dev + Track B, GitHub for code, Kaggle for GPU training + data/checkpoint storage.**

---

## 0. The one rule that prevents most failures

Three places, three jobs — never blur them:

| Place | Holds | Never holds |
|-------|-------|-------------|
| **Local laptop** | code (edited in Antigravity), Track B training, small tests, orchestration | heavy Track A training (1650 too small) |
| **GitHub** | source code only — the single source of truth | data, FITS, checkpoints, model weights |
| **Kaggle** | GPU compute + **Kaggle Datasets** for event-window data and checkpoints | the authoritative code (it pulls code from GitHub each run) |

If you ever feel tempted to `git add` a `.fits` or a `.pth`, stop — that file belongs in a Kaggle Dataset, not the repo.

---

## 1. Repository layout (runs identically local and on Kaggle)

```
solar-ar-forecast/
├── src/
│   ├── config.py          # detects local vs Kaggle → sets paths + device
│   ├── data/              # drms/Fido fetch, co-align, normalize
│   ├── seg/               # unet / surya / sam2 train + inference
│   ├── track/             # instance split + temporal tracking
│   ├── features/          # mask→AIA projection + feature extraction
│   ├── forecast/          # holt-winters, lstm, ensemble  (Track B)
│   └── eval/              # validation harness: splits, TSS/HSS/BSS
├── configs/               # one yaml per experiment
│   ├── unet.yaml
│   ├── surya.yaml
│   └── lstm_swansf.yaml
├── kaggle/
│   ├── kernel-metadata.json   # kaggle kernel config (gpu/internet/datasets)
│   ├── run_dataprep.ipynb     # thin launcher: fetch data → save dataset
│   └── run_train.ipynb        # thin launcher: clone repo → run a module
├── requirements.txt
├── .gitignore             # data/  *.fits  *.pth  *.h5  checkpoints/  outputs/
└── .env.example           # template for local secrets (never commit .env)
```

The Kaggle notebooks stay **thin launchers** — they clone the repo and call a module. All real logic lives in `src/`, version-controlled, so the same code path runs both places.

---

## 2. `config.py` — environment detection (the key to "write once, run both")

```python
import os
from pathlib import Path
import torch

ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input")          # read-only attached datasets
    OUT_DIR  = Path("/kaggle/working")        # persisted on "Save Version"
    CKPT_IN  = Path("/kaggle/input/solar-ckpts")  # prev checkpoints (if attached)
else:
    DATA_DIR = Path("./data")
    OUT_DIR  = Path("./outputs")
    CKPT_IN  = Path("./outputs/checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Guardrail: heavy seg training only where there's a real GPU
def assert_can_train_seg():
    if not ON_KAGGLE and torch.cuda.get_device_properties(0).total_memory < 8e9:
        raise RuntimeError("Seg training needs Kaggle (16GB) — 1650 is dev only.")
```

Everything downstream reads `DATA_DIR / OUT_DIR / DEVICE` from here — no hard-coded paths anywhere.

---

## 3. Data staging — fetch the data ON Kaggle, do not upload from home

This is the most important efficiency decision. Tens of GB over a home connection is painful, and Kaggle Datasets are the persistent store anyway — so let a Kaggle kernel with internet ON do the download.

`run_dataprep.ipynb` (run once per data-scope change):
1. Enable **GPU off, Internet on** for this kernel (it's I/O-bound, not GPU).
2. Put the JSOC-registered email in **Kaggle Secrets**, read it in the notebook.
3. Clone the repo, run `python -m src.data.fetch --config configs/data_v1.yaml` → writes 1024² FITS + a manifest CSV to `/kaggle/working`.
4. **Save Version** → "Save & Run All". The output becomes a Dataset, e.g. `solar-event-windows-v1`.
5. Every training kernel later attaches that dataset read-only at `/kaggle/input/solar-event-windows-v1`.

Result: data is downloaded once, lives on Kaggle, and never travels over your home link again.

---

## 4. Code sync — GitHub → Kaggle

Inside any Kaggle kernel, pull the exact code you want to run:

```python
# public repo
!git clone --depth 1 https://github.com/<you>/solar-ar-forecast.git
%cd solar-ar-forecast
!pip install -q -r requirements.txt
```

For a **private** repo, store a GitHub fine-grained PAT as a Kaggle Secret and clone with it (never paste the token in the notebook):

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_PAT")
!git clone --depth 1 https://{tok}@github.com/<you>/solar-ar-forecast.git
```

For reproducibility, pin and log the commit you ran:
```python
!git -C solar-ar-forecast rev-parse HEAD   # log this SHA in your results
```

---

## 5. Training kernel pattern

`kaggle/kernel-metadata.json` (created/pushed via the Kaggle CLI):
```json
{
  "id": "<you>/solar-train",
  "title": "solar-train",
  "code_file": "run_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<you>/solar-event-windows-v1", "<you>/solar-ckpts"]
}
```

`run_train.ipynb` body (thin):
```python
!git clone --depth 1 https://github.com/<you>/solar-ar-forecast.git
%cd solar-ar-forecast && pip install -q -r requirements.txt
!python -m src.seg.train --config configs/unet.yaml   # or surya.yaml
```

Outputs (checkpoints, metrics.json, figures) land in `/kaggle/working` and persist when you Save Version.

---

## 6. Checkpoint & resume — mandatory (Kaggle sessions cap ~12 h and can drop)

Build every training loop to survive a kill:

```python
ckpt = CKPT_IN / "latest.pth"
start_epoch = 0
if ckpt.exists():
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"]); optim.load_state_dict(state["optim"])
    start_epoch = state["epoch"] + 1

for epoch in range(start_epoch, n_epochs):
    train_one_epoch(...)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "epoch": epoch}, OUT_DIR / "latest.pth")   # every epoch
    if time_left() < 20*60:            # ~20 min before the 12h cap
        break                          # stop gracefully, Save Version
```

Loop: run → Save Version (output becomes `solar-ckpts`) → next run attaches `solar-ckpts` as input → resumes. Keep a rolling `latest.pth` plus periodic milestone copies.

---

## 7. Pulling results back to local

Use the Kaggle CLI from your local terminal (or let an Antigravity agent run it in the host shell):

```bash
kaggle kernels output <you>/solar-train -p ./outputs/kaggle_pull/
```

Small artifacts (metrics tables, reliability plots, run SHA) **can** be committed to GitHub for tracking. Large artifacts (checkpoints, FITS) stay on Kaggle — pull them locally only when you need to inspect, and keep them gitignored.

---

## 8. Automating the loop from Antigravity

Antigravity agents edit code and commit locally, but they can also drive the remote run headlessly through the Kaggle CLI in the host terminal:

```bash
# 1. agent edits src/, then:
git add -A && git commit -m "tweak unet aug" && git push
# 2. trigger a fresh Kaggle run of the pushed code:
kaggle kernels push -p ./kaggle/
# 3. poll until done:
kaggle kernels status <you>/solar-train
# 4. pull results for the agent to verify:
kaggle kernels output <you>/solar-train -p ./outputs/kaggle_pull/
```

Prereq: `pip install kaggle` and put the API token at `~/.kaggle/kaggle.json` (chmod 600). This is the closest you get to "Antigravity runs the GPU job" — it orchestrates Kaggle without you touching the web UI.

---

## 9. Track B stays fully local — no Kaggle in the loop

SWAN-SF + LSTM/Holt-Winters is tabular and light:
1. Download SWAN-SF once to `./data` locally.
2. `python -m src.forecast.train --config configs/lstm_swansf.yaml` on the 1650 (or CPU).
3. Commit the validated model summary + metrics to GitHub.

This is why Track B de-risks independently: it never depends on Kaggle quota, GitHub, or the sync loop.

---

## 10. Secrets & guardrails checklist

- Secrets live in **Kaggle Secrets** (JSOC email, GH_PAT) and a local **`.env`** — never in code or notebooks. Commit only `.env.example`.
- `.gitignore` must cover `data/  outputs/  *.fits  *.pth  *.h5  checkpoints/`. Consider a pre-commit hook that rejects files >50 MB.
- Kaggle **Internet must be ON** for the dataprep kernel (drms) and any `git clone`.
- Budget the **~30 GPU-hr/week**: dev and Track B locally; spend Kaggle GPU only on real Track A training; Surya runs are the expensive ones — schedule them.
- Log the **git commit SHA** in every Kaggle run's output for reproducibility.
- If Chalawan/ADA opens up later: point training there over SSH; nothing else in this runbook changes except where step 5 executes.
