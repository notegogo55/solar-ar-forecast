"""Kaggle script kernel -- Track A / A1 data acquisition.

Settings: GPU off, Internet ON (drms needs JSOC access).
Push via: kaggle kernels push -p ./kaggle/dataprep/
  (copy this file + a kernel-metadata.json to kaggle/dataprep/ and set id/title)

After "Save Version" the FITS files + manifest land in /kaggle/working and become
the Dataset named in configs/data_v1.yaml -> out_dataset.

Secrets required (set in Kaggle Secrets before running):
  JSOC_EMAIL  -- your registered JSOC email
  GH_PAT      -- GitHub fine-grained PAT (if repo is private)
"""
import os
import shutil
import subprocess

# ── 1. Pull source code ────────────────────────────────────────────────────
REPO = "https://github.com/notegogo55/solar-ar-forecast.git"

# Always start from /kaggle/working to avoid nested repo/ dirs across runs.
os.chdir("/kaggle/working")
shutil.rmtree("repo", ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, "repo"], check=True)
os.chdir("/kaggle/working/repo")

sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
print(f"git SHA: {sha}")

# ── 2. Install dependencies ────────────────────────────────────────────────
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

# ── 3. JSOC email from Kaggle Secrets (falls back to env var when local) ──
try:
    from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
    os.environ["JSOC_EMAIL"] = UserSecretsClient().get_secret("JSOC_EMAIL")
except ImportError:
    # Not on Kaggle -- JSOC_EMAIL must already be set in the environment.
    if not os.environ.get("JSOC_EMAIL"):
        raise RuntimeError(
            "Set JSOC_EMAIL env var (register at jsoc.stanford.edu first)"
        )

# ── 4. Fetch (dry-run first; remove --dry-run for the full window) ─────────
# Dry-run: 1 day + AIA 171 only -- validates the export path and manifest
# schema without waiting for the full ~13-day window.
# Once dry-run passes, re-run without --dry-run and "Save Version".
subprocess.run(
    ["python", "-u", "-m", "src.data.fetch",
     "--config", "configs/data_v1.yaml",
     "--dry-run"],                        # remove this line for the full fetch
    check=True,
)

# ── 5. Done -- outputs in /kaggle/working ─────────────────────────────────
# "Save Version" -> outputs become Dataset solar-event-windows-v1.
# Every training kernel attaches that dataset read-only at
# /kaggle/input/solar-event-windows-v1.
