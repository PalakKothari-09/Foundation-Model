import io
import random

import pytest

from minilm.tokenizer import Tokenizer, train_bpe, train_bpe_from_counts
from minilm.tokenizer.bpe import count_pretokens_text
from minilm.tokenizer.tokenizer import load_vocab_merges, save_vocab_merges

EOT = "<|endoftext|>"


# ---------------------------------------------------------------- training
def test_hand_worked_bpe_example():
    # low x5, lower x2, widest x3, newest x6 (split on spaces only -> counts given directly)
    counts = {b"low": 5, b"lower": 2, b"widest": 3, b"newest": 6}
    vocab, merges = train_bpe_from_counts(counts, 256 + 1 + 6, [EOT])
    assert merges == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    assert len(vocab) == 263


def test_tie_break_prefers_larger_pair():
    # pairs (a,b) and (b,c) both appear once: the larger tuple (b,c) must be merged first
    vocab, merges = train_bpe_from_counts({b"abc": 1}, 256 + 1 + 1, [EOT])
    assert merges == [(b"b", b"c")]


def test_vocab_size_and_unique_ids(tmp_path):
    p = tmp_path / "t.txt"
    rnd = random.Random(1)
    words = "the cat sat on a mat while dog ran far away and little birds sing songs about happy sunny days".split()
    p.write_text("".join(" ".join(rnd.choice(words) for _ in range(60)) + EOT for _ in range(50)))
    vocab, merges = train_bpe(str(p), 300, [EOT])
    assert len(vocab) == 300
    assert len(set(vocab.keys())) == 300
    assert len(set(vocab.values())) == 300  # no duplicate tokens either
    assert len(merges) == 300 - 257
    assert vocab[0] == EOT.encode()


def test_no_merge_crosses_story_boundary(tmp_path):
    p = tmp_path / "t.txt"
    # story A always ends with "xq" and story B always starts with "zj"; "qz"/"xqzj" must never be learned
    p.write_text(("hello xq" + EOT + "zj world" + EOT) * 100)
    vocab, _ = train_bpe(str(p), 400, [EOT])
    for tok in vocab.values():
        if tok == EOT.encode():
            continue
        assert EOT.encode() not in tok
        assert b"qz" not in tok and b"xqzj" not in tok


def test_special_tokens_not_counted_as_pairs():
    c = count_pretokens_text("ab" + EOT + "ab", [EOT])
    assert b"<|endoftext|>" not in c and c[b"ab"] == 2


def test_multiprocess_matches_single(tmp_path):
    p = tmp_path / "t.txt"
    story = "Once upon a time there was a little girl named Lily. She loved to play outside. "
    p.write_text("".join(story * (i % 5 + 1) + EOT for i in range(200)))
    a = train_bpe(str(p), 320, [EOT], num_workers=1)
    b = train_bpe(str(p), 320, [EOT], num_workers=2)
    assert a == b


# ---------------------------------------------------------------- encoding
def tiny_tokenizer(specials=None):
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    return Tokenizer(vocab, merges, specials)


def test_hand_worked_encoding():
    assert tiny_tokenizer().encode("the cat ate") == [9, 7, 1, 5, 10, 3]


def test_special_token_added_to_end_of_vocab():
    tok = tiny_tokenizer([EOT])
    assert tok.vocab[11] == EOT.encode()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "t.txt"
    p.write_text(("The quick brown fox jumps over the lazy dog. Lily and Tom played in the park. " * 30 + EOT) * 20)
    vocab, merges = train_bpe(str(p), 400, [EOT])
    return Tokenizer(vocab, merges, [EOT])


@pytest.mark.parametrize("text", ["", "Hello, world!", "  leading and trailing  \n\n", "नमस्ते दुनिया", "emoji 😀🎉 ok", "a" + EOT + "b", "ünïcödé"])
def test_roundtrip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_endoftext_is_one_id_and_isolates(trained):
    eot_id = trained.bytes_to_id[EOT.encode()]
    ids = trained.encode("the lazy" + EOT + "dog")
    assert ids.count(eot_id) == 1
    i = ids.index(eot_id)
    assert trained.encode("the lazy") == ids[:i] and trained.encode("dog") == ids[i + 1:]
    assert trained.encode(EOT) == [eot_id]


def test_overlapping_special_tokens_prefer_longer():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|a|>", "<|a|><|a|>"])
    ids = tok.encode("<|a|><|a|>")
    assert ids == [tok.bytes_to_id[b"<|a|><|a|>"]]


def test_decode_invalid_utf8_gives_replacement(trained):
    assert "�" in trained.decode([trained.bytes_to_id[b"\xff"]])


def test_encode_iterable_matches_encode(trained, tmp_path):
    rnd = random.Random(0)
    words = ["the", " fox", " jumps", "  ", "\n\n", "  \n", EOT, " 😀", "नमस्ते", "!!", " dog"]
    text = "".join(rnd.choice(words) for _ in range(4000))
    p = tmp_path / "f.txt"
    p.write_text(text, encoding="utf-8")
    with open(p, encoding="utf-8") as f:
        streamed = list(trained.encode_iterable(f, chunk_chars=50))
    assert streamed == trained.encode(text)


def test_encode_iterable_small_pieces_and_split_special(trained):
    text = "the fox" + EOT + "the dog"
    pieces = [text[i:i + 3] for i in range(0, len(text), 3)]  # cuts the special token in the middle
    assert list(trained.encode_iterable(pieces, chunk_chars=1)) == trained.encode(text)


def test_save_load_roundtrip(trained, tmp_path):
    save_vocab_merges(trained.vocab, trained.merges, tmp_path / "v.json", tmp_path / "m.txt")
    vocab, merges = load_vocab_merges(tmp_path / "v.json", tmp_path / "m.txt")
    assert vocab == trained.vocab and merges == trained.merges
    t2 = Tokenizer.from_files(tmp_path / "v.json", tmp_path / "m.txt", [EOT])
    s = "Lily 😀 played" + EOT
    assert t2.encode(s) == trained.encode(s)
