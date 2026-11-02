import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from minilm.generate import generate, generate_ids, sample_next
from minilm.model import TransformerLM
from minilm.tokenizer import Tokenizer
from minilm.train import (build_optimizer, build_scheduler, clip_gradients, cosine_lr, fixed_eval_starts, get_batch,
                          lm_loss, load_checkpoint, save_checkpoint)


def tiny(vocab=64):
    return TransformerLM(vocab, 16, 32, 2, 4, 64, 10000.0)


def test_loss_matches_manual_neg_log_softmax():
    logits, targets = torch.randn(3, 5, 11), torch.randint(0, 11, (3, 5))
    manual = -torch.log_softmax(logits, -1).gather(-1, targets[..., None]).mean()
    assert torch.allclose(lm_loss(logits, targets), manual, atol=1e-6)


def test_optimizer_groups_cover_every_param_once():
    m = tiny()
    opt = build_optimizer(m, 1e-3)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim == 1 for p in no_decay["params"])
    assert opt.defaults["betas"] == (0.9, 0.95)
    # every RMSNorm weight is in the no-decay group
    norm_ids = {id(mod.weight) for mod in m.modules() if isinstance(mod, torch.nn.RMSNorm)}
    assert norm_ids == {id(p) for p in no_decay["params"]}


def test_cosine_schedule_hand_values():
    kw = dict(lr_max=1.0, lr_min=0.1, warmup_steps=10, cosine_end=110)
    assert cosine_lr(0, **kw) == 0.0
    assert cosine_lr(5, **kw) == pytest.approx(0.5)  # t/Tw * max
    assert cosine_lr(10, **kw) == pytest.approx(1.0)  # cos(0) = 1 -> max
    assert cosine_lr(60, **kw) == pytest.approx(0.55)  # halfway: min + 0.5*(max-min)
    assert cosine_lr(110, **kw) == pytest.approx(0.1)  # reaches min exactly at T_c
    assert cosine_lr(500, **kw) == pytest.approx(0.1)


def test_scheduler_follows_function():
    m = tiny()
    opt = build_optimizer(m, 1e-3)
    sched = build_scheduler(opt, 1e-3, 1e-4, 10, 100)
    seen = []
    for t in range(101):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    for t in (0, 5, 10, 55, 100):
        assert seen[t] == pytest.approx(cosine_lr(t, 1e-3, 1e-4, 10, 100))
    assert seen[100] == pytest.approx(1e-4)  # lowest value exactly at the last step


def test_clipping_bounds_grad_norm_and_returns_pre_clip_norm():
    m = tiny()
    x = torch.randint(0, 64, (4, 16))
    (lm_loss(m(x), x) * 1000).backward()  # huge gradients
    pre = clip_gradients(m, 1.0)
    post = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters()))
    assert pre > 1.0 and post <= 1.0 + 1e-4


def test_get_batch_shift_and_bounds():
    data = np.arange(1000, dtype=np.uint16)  # token == its index, so shifts are checkable
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = get_batch(data, 8, 16, "cpu", rng)
        assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
        assert torch.equal(y, x + 1)
        assert int(y.max()) <= 999
    # smallest dataset where only one window fits: still no overrun
    small = np.arange(17, dtype=np.uint16)
    x, y = get_batch(small, 4, 16, "cpu", rng)
    assert torch.equal(y, x + 1) and int(y.max()) == 16


def test_fixed_eval_batches_are_deterministic():
    data = np.arange(5000, dtype=np.uint16)
    a, b = fixed_eval_starts(data, 3, 4, 16), fixed_eval_starts(data, 3, 4, 16)
    assert all(np.array_equal(i, j) for i, j in zip(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_autocast_loss_close_to_fp32():
    m = tiny().cuda()
    x = torch.randint(0, 64, (4, 16)).cuda()
    fp32 = lm_loss(m(x), x).item()
    with torch.autocast("cuda", dtype=torch.float16):
        fp16 = lm_loss(m(x), x).item()
    assert abs(fp32 - fp16) < 0.02


def _train_steps(m, opt, sched, batches):
    losses = []
    for x, y in batches:
        loss = lm_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        clip_gradients(m, 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def test_checkpoint_resume_reproduces_losses(tmp_path):
    data = np.random.default_rng(0).integers(0, 64, 5000).astype(np.uint16)

    def batches(seed):
        rng = np.random.default_rng(seed)
        return [get_batch(data, 4, 16, "cpu", rng) for _ in range(5)]

    torch.manual_seed(0)
    m = tiny()
    opt = build_optimizer(m, 1e-3)
    sched = build_scheduler(opt, 1e-3, 1e-4, 3, 20)
    _train_steps(m, opt, sched, batches(1))  # warm the optimizer state
    save_checkpoint(tmp_path / "c.pt", m, opt, sched, None, 5)
    ref = _train_steps(m, opt, sched, batches(2))

    m2 = tiny()
    opt2 = build_optimizer(m2, 1e-3)
    sched2 = build_scheduler(opt2, 1e-3, 1e-4, 3, 20)
    step, _ = load_checkpoint(tmp_path / "c.pt", m2, opt2, sched2)
    assert step == 5
    again = _train_steps(m2, opt2, sched2, batches(2))
    assert np.allclose(ref, again, atol=1e-3)


# ---------------------------------------------------------------- generation
@pytest.fixture(scope="module")
def tok_and_model():
    vocab = {0: b"<|endoftext|>", **{i + 1: bytes([i]) for i in range(256)}}
    tok = Tokenizer(vocab, [], ["<|endoftext|>"])
    torch.manual_seed(0)
    return tok, TransformerLM(len(tok.vocab), 16, 32, 2, 4, 64, 10000.0)


def test_greedy_generation_is_deterministic(tok_and_model):
    tok, m = tok_and_model
    outs = {generate(m, tok, "Once upon", 20, temperature=0.0, top_p=1.0) for _ in range(3)}
    assert len(outs) == 1


def test_generation_handles_prompts_longer_than_context(tok_and_model):
    tok, m = tok_and_model
    new = generate_ids(m, tok, "a" * 40, 5, temperature=0.0)  # context_length is 16
    assert len(new) <= 5


def test_seeded_sampling_reproducible(tok_and_model):
    tok, m = tok_and_model
    a = generate_ids(m, tok, "hi", 15, 1.0, 0.9, torch.Generator().manual_seed(7))
    b = generate_ids(m, tok, "hi", 15, 1.0, 0.9, torch.Generator().manual_seed(7))
    assert a == b


def test_top_p_keeps_only_nucleus():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05]))
    g = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, 1.0, 0.7, g) for _ in range(300)}
    assert picks == {0, 1}  # 0.5 + 0.3 >= 0.7 -> smallest set is {0, 1}
    assert {sample_next(logits, 1.0, 0.4, g) for _ in range(100)} == {0}  # top token alone already exceeds p


def test_generation_stops_at_eot():
    vocab = {0: b"<|endoftext|>", 1: b"a"}
    tok = Tokenizer(vocab, [], ["<|endoftext|>"])

    class AlwaysEot(torch.nn.Module):
        context_length = 8

        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            out = torch.full((*x.shape, 2), -10.0)
            out[..., 0] = 10.0
            return out

    assert generate_ids(AlwaysEot(), tok, "a", 10, temperature=0.0) == []
