"""Train the 10k BPE tokenizer; report time, peak memory, longest token; optional cProfile (Question 1).

python scripts/train_tokenizer.py --input C:/slm-data/TinyStoriesV2-GPT4-train.txt --out tokenizer_files --workers 4
python scripts/train_tokenizer.py --input ... --profile      # single process, prints cProfile + pre-token vs merge split
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.tokenizer.bpe import count_pretokens, train_bpe_from_counts  # noqa: E402
from minilm.tokenizer.tokenizer import save_vocab_merges  # noqa: E402

EOT = "<|endoftext|>"


def peak_mem_mb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20 if os.name == "nt" else psutil.Process().memory_info().rss / 2**20
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="tokenizer_files")
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-bytes", type=int, default=None, help="only use the first N bytes (e.g. first quarter)")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    prof = cProfile.Profile() if a.profile else None
    if prof:
        prof.enable()
    t_a = time.time()
    counts = count_pretokens(a.input, [EOT], a.workers, a.max_bytes)
    t_pre = time.time() - t_a
    t_b = time.time()
    vocab, merges = train_bpe_from_counts(counts, a.vocab_size, [EOT])
    t_merge = time.time() - t_b
    if prof:
        prof.disable()
        pstats.Stats(prof).sort_stats("cumulative").print_stats(12)
    total = time.time() - t0

    os.makedirs(a.out, exist_ok=True)
    save_vocab_merges(vocab, merges, os.path.join(a.out, "vocab.json"), os.path.join(a.out, "merges.txt"))
    longest = max(vocab.values(), key=len)
    print(f"unique pre-tokens: {len(counts):,}")
    print(f"pre-tokenise+count: {t_pre:.1f}s | merges: {t_merge:.1f}s | total: {total:.1f}s")
    print(f"peak memory: {peak_mem_mb():.0f} MB")
    print(f"vocab size: {len(vocab)}  longest token: {longest!r} ({len(longest)} bytes)")


if __name__ == "__main__":
    main()
