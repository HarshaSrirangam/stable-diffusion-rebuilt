import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        tokens = self.layernorm1(tokens)

        b, seq_len, d_model = tokens.shape
        # (B, 77, 768)
        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)

        # (B, 77, 768) -> (B, 77, n_heads, d_k) -> (B, n_heads, 77, d_k)
        q = q.reshape(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # causal mask: a token shouldn't attend to tokens after it
        # triu because masked_fill operates on true values
        # triu(1) because diagonal itself should stay False (so invert makes it True)
        # implement by setting attn score to -inf so softmax = 0
        attn_scores = q @ k.transpose(-1, -2)
        mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_scores = F.softmax(attn_scores / (self.d_k**0.5), dim=-1)

        tokens = attn_scores @ v
        tokens = tokens.transpose(1, 2).reshape(b, seq_len, -1)
        tokens = self.proj_out(tokens)
        tokens += residual

        residual = tokens
        tokens = self.layernorm2(tokens)
        tokens = self.linear1(tokens)
        # QuickGELU
        tokens = tokens * torch.sigmoid(1.702 * tokens)
        tokens = self.linear2(tokens)

        return tokens + residual


class CLIP(nn.Module):
    """Frozen CLIP text encoder."""

    def __init__(
        self, vocab_size=49408, max_seq_len=77, d_model=768, n_heads=12, n_blocks=12
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_heads=n_heads, d_model=d_model) for _ in range(n_blocks)]
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, max_seq_len)

        Returns: (B, max_seq_len, d_model)
        """
        _, seq_len = tokens.shape
        tokens = tokens.to(torch.long)

        token_embeddings = self.token_embedding_table(tokens)
        position_ids = torch.arange(start=0, end=seq_len, device=tokens.device)
        position_embeddings = self.position_embedding_table(position_ids)
        tokens = token_embeddings + position_embeddings

        for block in self.transformer_blocks:
            tokens = block(tokens)

        return self.layernorm(tokens)
