"""Short pilot runs at several learning rates (300 steps each), then one comparison plot.

python scripts/lr_sweep.py --train-data data/train.npy --val-data data/valid.npy
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/train.npy")
    ap.add_argument("--val-data", default="data/valid.npy")
    ap.add_argument("--lrs", nargs="+", default=["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    dirs = []
    for lr in a.lrs:
        d = f"runs/lr_{lr}"
        dirs.append(d)
        # same seed for every run -> identical init and identical batches; only the LR differs
        subprocess.run([sys.executable, os.path.join(ROOT, "train.py"), "--train-data", a.train_data, "--val-data", a.val_data,
                        "--out-dir", d, "--lr", lr, "--max-steps", str(a.steps), "--eval-interval", "50",
                        "--log-interval", "5", "--ckpt-interval", "100000", "--seed", "0", *a.extra], check=False)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot_runs.py"), "--runs", *dirs, "--out", "writeup/lr_sweep.png",
                    "--col", "train_loss", "--ymax", "8", "--title", "LR pilots (300 steps)"], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot_runs.py"), "--runs", *dirs, "--out", "writeup/lr_sweep_gradnorm.png",
                    "--col", "grad_norm", "--title", "grad norm"], check=True)
    for d in dirs:
        with open(os.path.join(d, "log.csv")) as f:
            rows = [r.split(",") for r in f.read().strip().split("\n")[1:]]
        vals = [r for r in rows if r[2]]
        print(d, "final train", rows[-1][1], "final val", vals[-1][2] if vals else "-")


if __name__ == "__main__":
    main()
