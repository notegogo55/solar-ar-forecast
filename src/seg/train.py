"""Track A / A2 — train a segmentation model (unet | surya | sam2).

All three share ONE label set + ONE held-out split (D3/D4). IoU/Dice do NOT
gate the spine: the first model to clear 'good enough' unblocks tracking.
Surya/SAM2 train on Kaggle (16GB); U-Net is the cheap unblocker.

Usage:  python -m src.seg.train --config configs/unet.yaml
"""
import argparse, yaml
from src.config import DEVICE, OUT_DIR, CKPT_IN, assert_can_train_seg


def main(cfg):
    assert_can_train_seg()  # refuses to run heavy seg on the 1650
    # TODO: build model per cfg['model']; resume from CKPT_IN/latest.pth if present;
    #       checkpoint to OUT_DIR every epoch (Kaggle 12h cap — see runbook sec 6);
    #       score IoU/Dice on the shared held-out split.
    raise NotImplementedError("implement seg training — see workplan A2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
