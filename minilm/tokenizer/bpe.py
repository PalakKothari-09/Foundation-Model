"""Byte-level BPE training."""
import heapq
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)


def special_split_pattern(special_tokens):
    """Regex matching any special token (longest first, escaped), or None."""
    if not special_tokens:
        return None
    return "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))


def find_chunk_boundaries(path, num_chunks, split_token=b"<|endoftext|>", end=None):
    """Byte offsets that cut the file into ~num_chunks pieces, each cut placed at a split token."""
    size = os.path.getsize(path)
    if end is not None:
        size = min(size, end)
    if num_chunks <= 1:
        return [0, size]
    chunk = size // num_chunks
    bounds = [i * chunk for i in range(num_chunks)] + [size]
    with open(path, "rb") as f:
        for i in range(1, len(bounds) - 1):
            pos = bounds[i]
            f.seek(pos)
            while True:
                buf = f.read(4096)
                if not buf:
                    bounds[i] = size
                    break
                j = buf.find(split_token)
                if j != -1:
                    bounds[i] = pos + j
                    break
                pos += 4096
    return sorted(set(bounds))


def count_pretokens_text(text, special_tokens):
    pat = special_split_pattern(special_tokens)
    docs = regex.split(pat, text) if pat else [text]
    str_counts = Counter()
    for doc in docs:
        str_counts.update(PRETOKEN_RE.findall(doc))  # count as str first; encode only the unique keys
    counts = Counter()
    for s, c in str_counts.items():
        counts[s.encode("utf-8")] += c
    return counts


def _count_chunk(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return count_pretokens_text(data.decode("utf-8", errors="ignore"), special_tokens)


def count_pretokens(path, special_tokens, num_workers=1, max_bytes=None):
    """Count pre-tokens (as bytes) in a file, optionally using several processes."""
    split = special_tokens[0].encode() if special_tokens else b"\n"
    bounds = find_chunk_boundaries(path, max(num_workers * 4, 1) if num_workers > 1 else 1, split, max_bytes)
    jobs = [(path, s, e, special_tokens) for s, e in zip(bounds[:-1], bounds[1:])]
    total = Counter()
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for c in pool.imap_unordered(_count_chunk, jobs):
                total.update(c)
    else:
        for j in jobs:
            total.update(_count_chunk(j))
    return total


class _Rev:
    """Heap key wrapper so that, among equal counts, the *larger* pair pops first."""
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def train_bpe_from_counts(counts, vocab_size, special_tokens):
    """BPE merges from a {pretoken_bytes: count} table.

    Ids: special tokens first, then the 256 bytes, then one new id per merge.
    """
    vocab = {}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    merges = []

    words, freqs = [], []
    for tok, c in counts.items():
        words.append([bytes([b]) for b in tok])
        freqs.append(c)

    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[i]
            pair_words[p].add(i)
    heap = [(-c, _Rev(p), p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size:
        best = None
        while heap:
            negc, _, p = heapq.heappop(heap)
            if pair_counts.get(p, 0) == -negc and -negc > 0:
                best = p
                break
        if best is None:
            break  # nothing left to merge
        a, b = best
        new = a + b
        merges.append(best)
        vocab[len(vocab)] = new

        changed = set()
        for i in list(pair_words[best]):
            w = words[i]
            if not any(w[j] == a and w[j + 1] == b for j in range(len(w) - 1)):
                continue  # stale entry
            f = freqs[i]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                changed.add(p)
            nw, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    nw.append(new)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            words[i] = nw
            for p in zip(nw, nw[1:]):
                pair_counts[p] += f
                pair_words[p].add(i)
                changed.add(p)
        del pair_words[best]
        for p in changed:
            c = pair_counts[p]
            if c <= 0:
                del pair_counts[p]
            else:
                heapq.heappush(heap, (-c, _Rev(p), p))
    return vocab, merges


def train_bpe(input_path, vocab_size, special_tokens, num_workers=1, max_bytes=None):
    """Train byte-level BPE on a text file. Returns (vocab: {id: bytes}, merges: [(bytes, bytes)])."""
    counts = count_pretokens(input_path, special_tokens, num_workers, max_bytes)
    return train_bpe_from_counts(counts, vocab_size, special_tokens)
