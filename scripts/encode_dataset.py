"""Encode a TinyStories text file to a uint16 .npy (parallel over story-aligned chunks).

python scripts/encode_dataset.py --input X.txt --tokenizer-dir tokenizer_files --out data/train.npy --workers 4
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.tokenizer import Tokenizer  # noqa: E402
from minilm.tokenizer.bpe import find_chunk_boundaries  # noqa: E402

EOT = "<|endoftext|>"
_tok = None


def _init(tok_dir):
    global _tok
    _tok = Tokenizer.from_files(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"), [EOT])


def _encode_chunk(args):
    path, s, e = args
    with open(path, "rb") as f:
        f.seek(s)
        data = f.read(e - s)
    return np.array(_tok.encode(data.decode("utf-8", errors="ignore")), dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer-dir", default="tokenizer_files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    size = os.path.getsize(a.input)
    n_chunks = max(1, size // (8 << 20))  # ~8MB chunks, cut at <|endoftext|>
    bounds = find_chunk_boundaries(a.input, n_chunks, EOT.encode())
    jobs = [(a.input, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    t0 = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init, initargs=(a.tokenizer_dir,)) as pool:
            parts = list(pool.imap(_encode_chunk, jobs, chunksize=1))
    else:
        _init(a.tokenizer_dir)
        parts = [_encode_chunk(j) for j in jobs]
    arr = np.concatenate(parts)
    dt = time.time() - t0
    assert arr.dtype == np.uint16
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, arr)
    print(f"{a.input}: {size:,} bytes -> {len(arr):,} tokens ({size / len(arr):.3f} bytes/token)")
    print(f"time {dt:.1f}s with {a.workers} worker(s): {size / dt:,.0f} bytes/s (aggregate)")


if __name__ == "__main__":
    main()
