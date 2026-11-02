from torch import nn

from .layers import CausalSelfAttention, FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm block: z = x + Attn(Norm(x)); y = z + FFN(Norm(z))."""

    def __init__(self, d_model, num_heads, d_ff, rope_theta=10000.0, max_len=256):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, num_heads, rope_theta, max_len)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        z = x + self.attn(self.attn_norm(x))
        return z + self.ffn(self.ffn_norm(z))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0):
        super().__init__()
        self.context_length = context_length
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
