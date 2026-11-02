import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def naive_causal_attention(q, k, v):
    """softmax(Q K^T / sqrt(d_k)) V with an explicit causal mask. Works for (..., seq, d) inputs."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    n_q, n_k = scores.shape[-2:]
    mask = torch.ones(n_q, n_k, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class RoPE(nn.Module):
    """Rotary position embedding: rotates each pair (x[2k], x[2k+1]) at position i by angle i / theta^(2k/d)."""

    def __init__(self, theta, d_head, max_len):
        super().__init__()
        assert d_head % 2 == 0
        inv_freq = 1.0 / theta ** (torch.arange(0, d_head, 2).float() / d_head)  # (d/2,)
        angles = torch.arange(max_len).float()[:, None] * inv_freq[None, :]  # (max_len, d/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions=None):
        """x: (..., seq, d_head). positions: optional (..., seq) ints; defaults to 0..seq-1."""
        seq = x.shape[-2]
        if positions is None:
            cos, sin = self.cos[:seq], self.sin[:seq]
        else:
            cos, sin = self.cos[positions], self.sin[positions]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope_theta=10000.0, max_len=256, use_rope=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(rope_theta, d_model // num_heads, max_len) if use_rope else None

    def forward(self, x):
        h = self.num_heads
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=h)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=h)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=h)
        if self.rope is not None:  # queries and keys only, never values
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(rearrange(out, "... h s d -> ... s (h d)"))


class FeedForward(nn.Module):
    """SwiGLU: W2 (SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
