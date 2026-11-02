"""Resume check: full run vs (stop halfway + --resume). Plots both loss curves; they must coincide.

python scripts/resume_check.py --train-data data/valid.npy --steps 60 [--device cpu]
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train import parse_args, train  # noqa: E402


def losses(d):
    with open(os.path.join(d, "log.csv")) as f:
        return {int(r["step"]): float(r["train_loss"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="writeup/resume_check.png")
    a, extra = ap.parse_known_args()  # unknown flags are forwarded to train.py
    base = ["--train-data", a.train_data, "--val-data", a.train_data, "--max-steps", str(a.steps), "--lr", "1e-3",
            "--log-interval", "1", "--eval-interval", "1000", "--ckpt-interval", "1000", "--eval-batches", "2",
            "--batch-size", "4", *extra]
    for d in ("runs/resume_full", "runs/resume_split"):
        shutil.rmtree(d, ignore_errors=True)
    train(parse_args(base + ["--out-dir", "runs/resume_full"]))
    train(parse_args(base + ["--out-dir", "runs/resume_split", "--stop-after", str(a.steps // 2)]))
    train(parse_args(base + ["--out-dir", "runs/resume_split", "--resume"]))
    full, split = losses("runs/resume_full"), losses("runs/resume_split")
    diff = max(abs(full[s] - split[s]) for s in full)
    print(f"max |loss difference| over {len(full)} steps: {diff:.2e}")
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*sorted(full.items())), label="uninterrupted", lw=3, alpha=0.5)
    plt.plot(*zip(*sorted(split.items())), "--", label=f"stopped at {a.steps // 2} + resumed")
    plt.axvline(a.steps // 2, color="gray", ls=":")
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title(f"Resume check (max diff {diff:.1e})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(a.out, dpi=130)
    assert diff < 1e-3, "resume diverged"


if __name__ == "__main__":
    main()
