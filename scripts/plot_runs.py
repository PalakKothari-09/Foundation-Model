"""Plot log.csv files.

python scripts/plot_runs.py --runs runs/lr_1e-4 runs/lr_3e-4 ... --out writeup/lr_sweep.png --col train_loss
python scripts/plot_runs.py --runs runs/main --out writeup/main_curve.png --col val_loss --time    # steps + time panels
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(os.path.join(path, "log.csv")) as f:
        return list(csv.DictReader(f))


def series(rows, col, x):
    pts = [(float(r[x]), float(r[col])) for r in rows if r[col] not in ("", None)]
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--col", default="train_loss", help="train_loss | val_loss | grad_norm | lr")
    ap.add_argument("--also-val", action="store_true", help="draw val_loss as dashed lines too")
    ap.add_argument("--time", action="store_true", help="add a second panel against elapsed seconds")
    ap.add_argument("--ymax", type=float)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(r.rstrip("/\\")) for r in a.runs]
    xs = [("step", "step")] + ([("elapsed_s", "time (s)")] if a.time else [])
    fig, axes = plt.subplots(1, len(xs), figsize=(6.2 * len(xs), 4), squeeze=False)
    for ax, (xcol, xlabel) in zip(axes[0], xs):
        for i, (r, lab) in enumerate(zip(a.runs, labels)):
            rows = read(r)
            color = f"C{i}"
            x, y = series(rows, a.col, xcol)
            ax.plot(x, y, label=f"{lab}", color=color)
            if a.also_val:
                x, y = series(rows, "val_loss", xcol)
                ax.plot(x, y, "--", marker="o", ms=3, color=color, label=f"{lab} (val)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(a.col)
        if a.ymax:
            ax.set_ylim(top=a.ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
