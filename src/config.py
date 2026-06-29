"""Single source of truth for paths + device. Import this everywhere —
never hard-code a path. Detects whether we're running locally (GTX 1650,
dev + Track B) or on Kaggle (P100/T4, Track A training).
"""
import os
from pathlib import Path

try:
    import torch
    _HAS_TORCH = True
except Exception:  # torch may be absent in a pure-data env
    _HAS_TORCH = False

ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input")                 # read-only attached datasets
    OUT_DIR = Path("/kaggle/working")                # persisted on "Save Version"
    CKPT_IN = Path("/kaggle/input/solar-ckpts")      # previous checkpoints (if attached)
else:
    DATA_DIR = Path("./data")
    OUT_DIR = Path("./outputs")
    CKPT_IN = Path("./outputs/checkpoints")

OUT_DIR.mkdir(parents=True, exist_ok=True)

if _HAS_TORCH:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"


def assert_can_train_seg():
    """Refuse to run heavy segmentation training on the 4GB 1650.
    Surya/SAM2/U-Net training belongs on Kaggle (16GB). See AGENTS.md."""
    if ON_KAGGLE:
        return
    if not _HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU — run segmentation training on Kaggle.")
    vram = torch.cuda.get_device_properties(0).total_memory
    if vram < 8e9:
        raise RuntimeError(
            f"Local GPU has {vram/1e9:.1f}GB — too small for seg training. "
            "Push to Kaggle (see docs/runbook-kaggle.md). The 1650 is dev + Track B only."
        )
