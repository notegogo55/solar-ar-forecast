"""Track B — Holt-Winters + LSTM ensemble forecaster.

Develop & VALIDATE on SWAN-SF first (no images needed), then swap in real
extracted features (D9). Runs LOCAL on the 1650/CPU — does not touch Kaggle.
Enforces the strict validation protocol (D8) via src.eval.

Usage:  python -m src.forecast.train --config configs/lstm_swansf.yaml
"""
import argparse, yaml
from src.config import DEVICE, OUT_DIR


def main(cfg):
    # TODO: load SWAN-SF; build HW baseline + LSTM; ensemble.
    # TODO: split via src.eval.splits (time-block + whole-AR), resample TRAIN only.
    # TODO: report TSS/HSS/BSS + reliability via src.eval.metrics on the
    #       natural-base-rate held-out window (evaluated once).
    raise NotImplementedError("implement Track B forecaster — see workplan Track B")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
