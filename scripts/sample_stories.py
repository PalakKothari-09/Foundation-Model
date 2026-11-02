"""Generate the three required stories (>= 256 tokens, "Once upon a time", two temperature/top-p settings).

python scripts/sample_stories.py --model-dir release --device cuda > writeup/stories.txt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import load_model  # noqa: E402
from minilm.generate import generate_ids  # noqa: E402

SETTINGS = [(0.7, 0.9, [1, 2]), (1.0, 0.95, [3])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    a = ap.parse_args()
    model, tok = load_model(a.model_dir, a.device)
    for temp, top_p, seeds in SETTINGS:
        for seed in seeds:
            g = torch.Generator(device=a.device).manual_seed(seed)
            # <|endoftext|> would end the story early; require >= 256 tokens by retrying other seeds if needed
            for attempt in range(20):
                new = generate_ids(model, tok, "Once upon a time", a.max_new_tokens, temp, top_p, g)
                if len(new) >= 256:
                    break
            print(f"=== temperature={temp} top_p={top_p} seed={seed} tokens={len(new)} ===")
            print("Once upon a time" + tok.decode(new))
            print()


if __name__ == "__main__":
    main()
