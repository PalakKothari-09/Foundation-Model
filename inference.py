"""Generate a story from a trained release folder (CPU is fine)."""
import argparse
import json
import os
import sys
import time

import torch

from minilm.generate import generate_ids
from minilm.model import TransformerLM
from minilm.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def load_model(model_dir, device="cpu"):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = TransformerLM(cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
                          cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"])
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    tok = Tokenizer.from_files(os.path.join(model_dir, "vocab.json"), os.path.join(model_dir, "merges.txt"), [EOT])
    return model.to(device).eval(), tok


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap =argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    model, tok = load_model(a.model_dir, a.device)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    t0 = time.perf_counter()
    new = generate_ids(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_p, gen)
    dt = time.perf_counter() - t0
    print(a.prompt + tok.decode(new))
    print(f"\n[{len(new)} tokens generated, {len(new) / dt:.1f} tokens/s on {a.device}]")


if __name__ == "__main__":
    main()
