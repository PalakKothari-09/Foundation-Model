"""Training script. Every setting is a command-line flag (or a key in --config JSON).

Example:
  python train.py --train-data data/train.npy --val-data data/valid.npy --out-dir runs/main \
      --lr 1e-3 --max-steps 5000
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from minilm.model import TransformerLM
from minilm.train import (build_optimizer, build_scheduler, clip_gradients, evaluate, fixed_eval_starts,
                          get_batch, lm_loss, load_checkpoint, load_tokens, save_checkpoint)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="JSON file whose keys override the defaults (CLI flags win)")
    p.add_argument("--train-data", default="data/train.npy")
    p.add_argument("--val-data", default="data/valid.npy")
    p.add_argument("--out-dir", default="runs/run")
    # model
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1, help="alpha_min = frac * lr")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", choices=["fp16", "fp32"], default=None, help="default: fp16 on CUDA, else fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # logging / checkpoints / eval
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from a checkpoint path (no value: <out-dir>/ckpt.pt)")
    p.add_argument("--stop-after", type=int, default=None, help="stop early at this step (schedule unchanged)")
    p.add_argument("--overfit-one-batch", action="store_true", help="sanity check: train on a single fixed batch")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default=None)

    args, _ = p.parse_known_args(argv)
    if args.config:
        with open(args.config) as f:
            p.set_defaults(**{k.replace("-", "_"): v for k, v in json.load(f).items()})
    return p.parse_args(argv)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.precision or ("fp16" if device == "cuda" else "fp32")) == "fp16" and device == "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_data, val_data = load_tokens(args.train_data), load_tokens(args.val_data)
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                          args.num_heads, args.d_ff, args.rope_theta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} amp={use_amp} params={n_params:,}")

    opt = build_optimizer(model, args.lr, (args.beta1, args.beta2), args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.max_steps))
    sched = build_scheduler(opt, args.lr, args.min_lr_frac * args.lr, warmup, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fwd = torch.compile(model) if args.compile else model

    step = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    log_path = os.path.join(args.out_dir, "log.csv")
    resume_path = ckpt_path if args.resume == "auto" else args.resume
    if resume_path:
        step, extra = load_checkpoint(resume_path, model, opt, sched, scaler, map_location=device)
        if "rng" in extra:
            rng.bit_generator.state = extra["rng"]
        print(f"resumed from {resume_path} at step {step}")
    elif os.path.exists(log_path):
        os.remove(log_path)

    val_starts = fixed_eval_starts(val_data, args.eval_batches, args.batch_size, args.context_length)
    fixed_batch = get_batch(train_data, args.batch_size, args.context_length, device, rng) if args.overfit_one_batch else None

    log_new = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if log_new:
        log.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s"])
    if args.wandb:
        import wandb
        wandb.init(project="minilm", config=vars(args), dir=args.out_dir)

    def checkpoint():
        save_checkpoint(ckpt_path, model, opt, sched, scaler, step, {"rng": rng.bit_generator.state})

    end = min(args.max_steps, args.stop_after) if args.stop_after else args.max_steps
    t0 = time.time()
    model.train()
    while step < end:
        x, y = fixed_batch if fixed_batch is not None else get_batch(train_data, args.batch_size, args.context_length, device, rng)
        lr = opt.param_groups[0]["lr"]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = lm_loss(fwd(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # unscale BEFORE clipping so the clip threshold is in true-gradient units
        gnorm = clip_gradients(model, args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        val = ""
        if step % args.eval_interval == 0 or step == end:
            val = evaluate(model, val_data, val_starts, args.context_length, device, use_amp)
        if step % args.log_interval == 0 or val != "" or step == 1:
            tl, gn = loss.item(), float(gnorm)
            if not math.isfinite(tl):
                raise RuntimeError(f"non-finite loss at step {step}")
            row = [step, f"{tl:.5f}", f"{val:.5f}" if val != "" else "", f"{lr:.3e}", f"{gn:.4f}", f"{time.time() - t0:.1f}"]
            log.writerow(row)
            log_f.flush()
            if args.wandb:
                wandb.log({"train_loss": tl, "val_loss": val or None, "lr": lr, "grad_norm": gn}, step=step)
            if step % (args.log_interval * 10) == 0 or val != "":
                print(" ".join(map(str, row)))
        if step % args.ckpt_interval == 0:
            checkpoint()
    checkpoint()
    log_f.close()
    return model


if __name__ == "__main__":
    train(parse_args())
