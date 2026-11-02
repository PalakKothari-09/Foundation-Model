"""Time training steps with and without torch.compile (run on the GPU).

python scripts/bench_compile.py --steps 200
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.model import TransformerLM  # noqa: E402
from minilm.train import build_optimizer, clip_gradients, lm_loss  # noqa: E402


def bench(compile_, steps, device, amp):
    torch.manual_seed(0)
    m = TransformerLM(10000, 256, 512, 4, 16, 1344, 10000.0).to(device)
    f = torch.compile(m) if compile_ else m
    opt = build_optimizer(m, 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randint(0, 10000, (32, 256), device=device)

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = lm_loss(f(x), x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_gradients(m)
        scaler.step(opt)
        scaler.update()

    t = time.time()
    for _ in range(10):  # warm-up (includes compilation)
        step()
    torch.cuda.synchronize()
    warm = time.time() - t
    t = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return warm, (time.time() - t) / steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    res = {}
    for c in (False, True):
        warm, per = bench(c, a.steps, "cuda", not a.fp32)
        res[c] = per
        print(f"compile={c}: warm-up {warm:.1f}s, {per * 1000:.1f} ms/step")
    print(f"speed-up: {res[False] / res[True]:.2f}x")
