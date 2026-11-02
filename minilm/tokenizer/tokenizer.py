"""BPE Tokenizer: encode / decode / streaming encode, plus (de)serialisation."""
import json

import regex

from .bpe import PRETOKEN_RE, special_split_pattern


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode char map (used only to store tokens in JSON/text)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_B2U = bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def _to_str(b):
    return "".join(_B2U[x] for x in b)


def _to_bytes(s):
    return bytes(_U2B[c] for c in s)


def save_vocab_merges(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({_to_str(b): i for i, b in vocab.items()}, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_to_str(a)} {_to_str(b)}\n")


def load_vocab_merges(vocab_path, merges_path):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {i: _to_bytes(s) for s, i in json.load(f).items()}
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                a, b = line.split(" ")
                merges.append((_to_bytes(a), _to_bytes(b)))
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.bytes_to_id = {b: i for i, b in self.vocab.items()}
        for t in self.special_tokens:
            tb = t.encode("utf-8")
            if tb not in self.bytes_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = tb
                self.bytes_to_id[tb] = new_id
        self.rank = {pair: r for r, pair in enumerate(self.merges)}
        pat = special_split_pattern(self.special_tokens)
        self._special_re = regex.compile(f"({pat})") if pat else None
        self._cache = {}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        vocab, merges = load_vocab_merges(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, tok):
        hit = self._cache.get(tok)
        if hit is not None:
            return hit
        parts = [bytes([b]) for b in tok]
        while len(parts) > 1:
            best, best_rank = None, None
            for pair in zip(parts, parts[1:]):
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            a, b = best
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        ids = [self.bytes_to_id[p] for p in parts]
        if len(self._cache) < 500_000:
            self._cache[tok] = ids
        return ids

    def _encode_plain(self, text):
        ids = []
        for m in PRETOKEN_RE.finditer(text):
            ids.extend(self._encode_pretoken(m.group(0).encode("utf-8")))
        return ids

    def encode(self, text):
        if self._special_re is None:
            return self._encode_plain(text)
        ids = []
        for piece in self._special_re.split(text):  # capturing group keeps the specials
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.bytes_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_plain(piece))
        return ids

    def encode_iterable(self, iterable, chunk_chars=1 << 16):
        """Yield token ids from an iterable of strings (e.g. an open text file) with bounded memory.

        Gives the same ids as encode(whole_text): the tail of the buffer that could still change
        (last pre-token, or a partial special token) is held back until more text arrives.
        """
        max_special = max((len(t) for t in self.special_tokens), default=0)
        buf = ""
        for piece in iterable:
            buf += piece
            if len(buf) < chunk_chars:
                continue
            done, buf = self._split_safe(buf, max_special)
            yield from self.encode(done)
        yield from self.encode(buf)

    def _split_safe(self, buf, max_special):
        """Split buf into (text that is safe to encode now, held-back tail)."""
        cut = len(buf)
        # hold back anything that could be the start of a special token
        for t in self.special_tokens:
            for k in range(min(len(t) - 1, len(buf)), 0, -1):
                if buf.endswith(t[:k]):
                    cut = min(cut, len(buf) - k)
                    break
        # hold back the last pre-token (it may extend / change with following text)
        start = 0
        if self._special_re is not None:
            last = None
            for m in self._special_re.finditer(buf):
                last = m
            if last is not None:
                start = last.end()
        last_start = None
        for m in PRETOKEN_RE.finditer(buf, start):
            last_start = m.start()
        if last_start is not None:
            cut = min(cut, last_start)
        else:
            cut = min(cut, start)
        cut = max(cut, 0)
        return buf[:cut], buf[cut:]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
