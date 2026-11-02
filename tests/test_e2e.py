"""train.py -> make_release.py -> inference.py on a toy corpus (CPU, seconds)."""
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from minilm.tokenizer import Tokenizer, train_bpe
from minilm.tokenizer.tokenizer import save_vocab_merges

ROOT = Path(__file__).resolve().parent.parent
EOT = "<|endoftext|>"
TINY = ["--vocab-size", "300", "--context-length", "32", "--d-model", "32", "--num-layers", "2", "--num-heads", "4",
        "--d-ff", "64", "--batch-size", "4"]


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def test_train_resume_release_inference(tmp_path):
    rnd = random.Random(0)
    words = "the cat sat on a mat while dog ran far away and little birds sing songs".split()
    text = "".join(" ".join(rnd.choice(words) for _ in range(80)) + EOT for _ in range(40))
    (tmp_path / "t.txt").write_text(text)
    vocab, merges = train_bpe(str(tmp_path / "t.txt"), 300, [EOT])
    tokd = tmp_path / "tok"
    tokd.mkdir()
    save_vocab_merges(vocab, merges, tokd / "vocab.json", tokd / "merges.txt")
    ids = np.array(Tokenizer(vocab, merges, [EOT]).encode(text), dtype=np.uint16)
    np.save(tmp_path / "d.npy", ids)

    common = ["train.py", "--train-data", tmp_path / "d.npy", "--val-data", tmp_path / "d.npy", "--out-dir", tmp_path / "run",
              "--eval-interval", "5", "--eval-batches", "2", "--log-interval", "1", "--lr", "3e-3", *TINY]
    run(*common, "--max-steps", "10", "--stop-after", "5")
    out = run(*common, "--max-steps", "10", "--resume")
    assert "resumed" in out.stdout and "at step 5" in out.stdout
    with open(tmp_path / "run" / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows][-1] == 10
    assert all(np.isfinite(float(r["grad_norm"])) for r in rows)
    val = [float(r["val_loss"]) for r in rows if r["val_loss"]]
    assert len(val) >= 2 and val[-1] < val[0]

    rel = tmp_path / "release"
    run("scripts/make_release.py", "--ckpt", tmp_path / "run" / "ckpt.pt", "--tokenizer-dir", tokd, "--out", rel)
    assert sorted(p.name for p in rel.iterdir()) == ["config.json", "merges.txt", "model.pt", "vocab.json"]
    assert json.loads((rel / "config.json").read_text())["d_model"] == 32
    cmd = ["inference.py", "--model-dir", rel, "--prompt", "the cat", "--max-new-tokens", "20", "--seed", "42", "--device", "cpu"]
    a, b = run(*cmd).stdout, run(*cmd).stdout
    story = lambda s: s.split("\n[")[0]
    assert story(a) == story(b) and "tokens/s" in a
