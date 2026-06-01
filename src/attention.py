import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # one big W for QKV
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=in_proj_bias) 
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias) 
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.head_dim = embed_dim // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (B,H*W,C) 
        # or
        # tokens: (B,77,768)

        input_shape = x.shape
        b, seq_len, _ = input_shape
        interim_shape = (b, seq_len, self.n_heads, self.head_dim)

        x = self.in_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=-1)

        # split into heads: x: (B,77,768) -> (B,77,8,96) -> (B,8,77,96)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # compute attention scores
        attn_scores = q @ k.transpose(-1, -2)

        # causal mask: a token shouldn't attend to tokens after it (used for CLIP)
        # triu because masked_fill operates on true values
        # triu(1) because diagonal itself should stay False (so invert makes it True)
        # implement by setting attn score to -inf so softmax = 0
        if causal_mask:
             mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
             attn_scores = attn_scores.masked_fill(mask, -torch.inf)


        # normalize by head dim and softmax
        attn_scores /= (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # re-embed tokens
        x = attn_scores @ v

        # reshape: (B,heads,seq_len, head_dim) -> (B, seq_len, heads, head_dim) -> (B, seq_len, embed_dim)
        x = x.transpose(1, 2)
        x = x.reshape(b, seq_len, -1)

        # allow model to mix information between heads
        x = self.out_proj(x)

        return x
