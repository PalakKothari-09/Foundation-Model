"""Numbers for the writeup: compression ratio, Q3, Q4, Q5-Q8, learning-rate warm-up.

python scripts/analysis.py --valid-text C:/slm-data/TinyStoriesV2-GPT4-valid.txt --tokenizer-dir tokenizer_files
"""
import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.model import RoPE, TransformerLM  # noqa: E402
from minilm.tokenizer import Tokenizer  # noqa: E402

EOT = "<|endoftext|>"


def compression(a):
    tok = Tokenizer.from_files(os.path.join(a.tokenizer_dir, "vocab.json"), os.path.join(a.tokenizer_dir, "merges.txt"), [EOT])
    with open(a.valid_text, encoding="utf-8") as f:
        stories = [s.strip() for s in f.read().split(EOT) if s.strip()]
    rnd = random.Random(0)
    sample = rnd.sample(stories, 10)
    nbytes = sum(len(s.encode()) for s in sample)
    ntok = sum(len(tok.encode(s)) for s in sample)
    print(f"[compression] 10 random valid stories: {nbytes} bytes / {ntok} tokens = {nbytes / ntok:.3f} bytes/token")


def q3():
    torch.manual_seed(0)
    d_k = 32
    q, k = torch.randn(1000, d_k), torch.randn(d_k, 16)  # unit-variance q,k entries -> var(q.k) = d_k
    raw = (q @ k)[:3]
    scaled = raw / math.sqrt(d_k)
    print(f"[Q3] score std raw={(q @ k).std():.2f} (~sqrt(32)={math.sqrt(32):.2f}), scaled={(raw / math.sqrt(d_k)).std():.2f}")
    for name, s in [("raw", raw), ("scaled", scaled)]:
        p = torch.softmax(s, -1)
        ent = -(p * p.log()).sum(-1)
        print(f"     {name:6s} max-prob per row={[round(x, 3) for x in p.max(-1).values.tolist()]} entropy={[round(x, 2) for x in ent.tolist()]} (uniform over 16 = {math.log(16):.2f})")
    # gradient of softmax saturates when one prob ~ 1
    s = (q @ k)[0].clone().requires_grad_()
    p = torch.softmax(s, -1)
    g = torch.autograd.functional.jacobian(lambda z: torch.softmax(z, -1), s).abs().max().item()
    s2 = ((q @ k)[0] / math.sqrt(d_k)).clone().requires_grad_()
    g2 = torch.autograd.functional.jacobian(lambda z: torch.softmax(z, -1), s2).abs().max().item()
    print(f"     max |d softmax/d score|: raw={g:.4f}, scaled={g2:.4f}")


def q4():
    torch.manual_seed(0)
    d, L = 64, 32
    rope = RoPE(10000.0, d, L)
    q, k = torch.randn(d), torch.randn(d)
    Q, K = rope(q.expand(L, d).contiguous()), rope(k.expand(L, d).contiguous())
    print("[Q4] q_i . k_j after RoPE (same q, k at every position):")
    for i, j in [(3, 1), (10, 8), (25, 23), (7, 2), (30, 25), (5, 5), (20, 20)]:
        print(f"     i={i:2d} j={j:2d} i-j={i - j}: {float(Q[i] @ K[j]):.6f}")
    print(f"     (without RoPE every pair gives {float(q @ k):.6f})")


def flops(V, L, d, layers, heads, dff):
    """Matmul FLOPs for one forward pass over L tokens (2mnp per matmul)."""
    dh = d // heads
    rows = {
        "q,k,v,o projections (per layer)": 4 * 2 * L * d * d,
        "attention QK^T (per layer)": 2 * heads * L * L * dh,
        "attention AV (per layer)": 2 * heads * L * L * dh,
        "SwiGLU w1,w3,w2 (per layer)": 3 * 2 * L * d * dff,
    }
    per_layer = sum(rows.values())
    head = 2 * L * d * V
    total = layers * per_layer + head
    return rows, per_layer, head, total


def q5_q7():
    cfg = dict(V=10_000, L=256, d=512, layers=4, heads=16, dff=1344)
    m = TransformerLM(cfg["V"], cfg["L"], cfg["d"], cfg["layers"], cfg["heads"], cfg["dff"], 10000.0)
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for p in m.embed.parameters()) + sum(p.numel() for p in m.lm_head.parameters())
    print(f"[Q5] total params {total:,}; embedding+output {emb:,} ({100 * emb / total:.1f}%); blocks {sum(p.numel() for p in m.blocks.parameters()):,}")
    for name, c in [("Q6 (ours)", cfg), ("Q7 (GPT-2 small)", dict(V=50_257, L=1024, d=768, layers=12, heads=12, dff=2048))]:
        rows, per_layer, head, tot = flops(**c)
        print(f"[{name}] total {tot / 1e9:.2f} GFLOPs / forward pass")
        for k, v in rows.items():
            print(f"     {k:34s} {v / 1e9:8.3f} G  x{c['layers']} = {c['layers'] * v / 1e9:8.3f} G ({100 * c['layers'] * v / tot:.1f}%)")
        print(f"     {'output projection (lm_head)':34s} {head / 1e9:8.3f} G ({100 * head / tot:.1f}%)")
    return total


def q8(total):
    mb = lambda n: n * 4 / 2**20
    print(f"[Q8] fp32: params {mb(total):.1f} MiB, grads {mb(total):.1f} MiB, AdamW m+v {mb(2 * total):.1f} MiB -> {mb(4 * total):.1f} MiB total")


def lr_warmup():
    print("[warm-up] SGD on mean(w^2), w = 5*randn(10,10); grad = 2w/100 -> update factor (1 - lr/50)")
    for lr in (1, 10, 100, 1000):
        torch.manual_seed(0)
        w = (5 * torch.randn(10, 10)).requires_grad_()
        opt = torch.optim.SGD([w], lr=lr)
        losses = []
        for _ in range(10):
            loss = (w ** 2).mean()
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"     lr={lr:5d}: " + ", ".join(f"{x:.3g}" for x in losses))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-text")
    ap.add_argument("--tokenizer-dir", default="tokenizer_files")
    a = ap.parse_args()
    if a.valid_text:
        compression(a)
    q3()
    q4()
    q8(q5_q7())
    lr_warmup()
