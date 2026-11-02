"""Export weights-only model.pt + config.json + tokenizer files into release/.

python scripts/make_release.py --ckpt runs/main/ckpt.pt --tokenizer-dir tokenizer_files --out release
"""
import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-args", help="args.json of the run (default: next to the checkpoint)")
    ap.add_argument("--tokenizer-dir", default="tokenizer_files")
    ap.add_argument("--out", default="release")
    a = ap.parse_args()

    run_args = a.run_args or os.path.join(os.path.dirname(a.ckpt), "args.json")
    with open(run_args) as f:
        r = json.load(f)
    cfg = {k: r[k] for k in ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"]}
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    os.makedirs(a.out, exist_ok=True)
    torch.save({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}, os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name in ["vocab.json", "merges.txt"]:
        shutil.copy(os.path.join(a.tokenizer_dir, name), os.path.join(a.out, name))
    print("wrote", a.out, "from step", ck["step"])


if __name__ == "__main__":
    main()
