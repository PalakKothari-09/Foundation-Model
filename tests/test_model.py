import math

import pytest
import torch
import torch.nn.functional as F

from minilm.model import (CausalSelfAttention, FeedForward, RoPE, TransformerBlock, TransformerLM,
                          naive_causal_attention)

torch.manual_seed(0)


@pytest.mark.parametrize("shape", [(5, 8), (2, 5, 8), (2, 3, 5, 8)])
def test_naive_attention_matches_sdpa(shape):
    q, k, v = (torch.randn(*shape) for _ in range(3))  # 2-D (seq,d) / 3-D (batch,seq,d) / 4-D (batch,heads,seq,d)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(naive_causal_attention(q, k, v), ref, atol=1e-5)


def test_naive_attention_is_actually_causal():
    q, k, v = (torch.randn(1, 6, 4) for _ in range(3))
    out = naive_causal_attention(q, k, v)
    assert torch.allclose(out[0, 0], v[0, 0], atol=1e-6)  # first token can only attend to itself


def rope_matrix_reference(theta, d, pos):
    """Block-diagonal rotation matrix R_pos built explicitly (slow, obviously-correct version)."""
    R = torch.zeros(d, d)
    for k in range(d // 2):
        ang = pos / theta ** (2 * k / d)
        c, s = math.cos(ang), math.sin(ang)
        R[2 * k, 2 * k], R[2 * k, 2 * k + 1] = c, -s
        R[2 * k + 1, 2 * k], R[2 * k + 1, 2 * k + 1] = s, c
    return R


def test_rope_matches_rotation_matrix():
    d, L = 16, 12
    rope = RoPE(10000.0, d, L)
    x = torch.randn(3, L, d)
    expected = torch.stack([torch.stack([rope_matrix_reference(10000.0, d, i) @ x[b, i] for i in range(L)]) for b in range(3)])
    assert torch.allclose(rope(x), expected, atol=1e-5)


def test_rope_buffers_not_in_state_dict():
    assert len(RoPE(10000.0, 8, 4).state_dict()) == 0


def test_rope_position_zero_is_identity_and_preserves_norm():
    rope = RoPE(10000.0, 8, 10)
    x = torch.randn(10, 8)
    y = rope(x)
    assert torch.allclose(y[0], x[0])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property():
    d, L = 32, 20
    rope = RoPE(10000.0, d, L)
    q, k = torch.randn(d), torch.randn(d)
    Q = rope(q.expand(L, d).contiguous())  # same q at every position
    K = rope(k.expand(L, d).contiguous())
    dots = lambda i, j: float(Q[i] @ K[j])
    assert abs(dots(5, 2) - dots(15, 12)) < 1e-4  # both i - j = 3
    assert abs(dots(3, 3) - dots(9, 9)) < 1e-4
    assert abs(dots(5, 2) - dots(5, 3)) > 1e-3  # different offset -> different value


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_shapes_with_extra_batch_dims(batch):
    x = torch.randn(*batch, 7, 32)
    assert CausalSelfAttention(32, 4, max_len=16)(x).shape == x.shape
    assert FeedForward(32, 64)(x).shape == x.shape
    assert TransformerBlock(32, 4, 64, max_len=16)(x).shape == x.shape
    assert RoPE(10000.0, 8, 16)(torch.randn(*batch, 7, 8)).shape == (*batch, 7, 8)


def test_attention_rope_only_on_q_and_k():
    # with identical q/k/v weights path, changing RoPE theta must change output (RoPE active),
    # but a single token (position 0, identity rotation) must be unaffected
    a1 = CausalSelfAttention(32, 4, rope_theta=10000.0, max_len=8)
    a2 = CausalSelfAttention(32, 4, rope_theta=10.0, max_len=8)
    a2.load_state_dict(a1.state_dict())
    x = torch.randn(1, 5, 32)
    assert not torch.allclose(a1(x), a2(x), atol=1e-5)
    assert torch.allclose(a1(x)[:, :1], a2(x)[:, :1], atol=1e-5)


def test_feedforward_is_swiglu():
    ff = FeedForward(8, 16)
    x = torch.randn(3, 8)
    expected = ff.w2.weight @ (F.silu(ff.w1.weight @ x[0]) * (ff.w3.weight @ x[0]))
    assert torch.allclose(ff(x)[0], expected, atol=1e-5)


def small_lm(vocab=100):
    return TransformerLM(vocab, 32, 64, 2, 4, 128, 10000.0)


def test_lm_logits_shape():
    lm = TransformerLM(10_000, 32, 64, 2, 4, 128, 10000.0)
    assert lm(torch.randint(0, 10_000, (2, 17))).shape == (2, 17, 10_000)


def test_model_cannot_see_the_future():
    lm = small_lm().eval()
    a = torch.randint(0, 100, (2, 20))
    b = a.clone()
    t = 9
    b[:, t + 1:] = torch.randint(0, 100, (2, 20 - t - 1))
    assert not torch.equal(a, b)
    with torch.no_grad():
        la, lb = lm(a), lm(b)
    assert torch.equal(la[:, :t + 1], lb[:, :t + 1])  # identical at <= t ...
    assert not torch.allclose(la[:, t + 1:], lb[:, t + 1:])  # ... and the test is not vacuous


def test_initial_loss_is_log_vocab():
    # With ~zero-information logits the softmax is near-uniform over V classes, so the cross
    # entropy of any target is about -log(1/V) = ln V (= ln 10000 ~ 9.21). A much different value
    # at init signals bad init scale or a bug in the loss/targets.
    V = 10_000
    lm = TransformerLM(V, 64, 128, 2, 4, 256, 10000.0)
    x = torch.randint(0, V, (8, 64))
    y = torch.randint(0, V, (8, 64))
    loss = F.cross_entropy(lm(x).reshape(-1, V), y.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.5


def test_param_count_of_assignment_config():
    lm = TransformerLM(10_000, 256, 512, 4, 16, 1344, 10000.0)
    total = sum(p.numel() for p in lm.parameters())
    per_layer = 4 * 512 * 512 + 3 * 512 * 1344 + 2 * 512
    expected = 2 * 10_000 * 512 + 4 * per_layer + 512
    assert total == expected == 22_696_448  # 10,240,000 (embed+head) + 12,455,936 (4 layers) + 512 (final norm)
    assert 4 * (4 * 512 * 512 + 3 * 512 * 1344) == 12_451_840  # ~12.5M in the transformer layers
