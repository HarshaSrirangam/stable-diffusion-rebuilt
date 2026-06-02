import torch
import torch.nn as nn
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """ Token and positional embedder. """
    def __init__(self, vocab_size: int, d_model: int, seq_length):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Parameter(torch.zeros(seq_length, d_model))

    def forward(self, tokens: torch.Tensor):
        assert tokens.shape[1] == 77, "max 77 tokens allowed"
        return self.token_embedding_table(tokens) + self.position_embeddings

class CLIPLayer(nn.Module):
    """ CLIP transformer block. """
    def __init__(self, n_head, d_model):
        super().__init__()
        self.attention = SelfAttention(n_head, d_model)
        self.linear_1 = nn.Linear(d_model, 4*d_model)
        self.linear_2 = nn.Linear(4*d_model, d_model)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.FloatTensor) -> torch.FloatTensor:
        assert tokens.shape[1:] == (77, 768), "embedding dim must be 768"
        # multi-head attention with residual connection
        residue = tokens
        tokens = self.layernorm_1(tokens)
        tokens = self.attention(tokens, causal_mask=True)
        tokens = tokens + residue

        # feed-forward network with residual connection
        residue = tokens
        tokens = self.layernorm_2(tokens)
        tokens = self.linear_1(tokens)
        tokens = tokens * torch.sigmoid(1.702 * tokens) # QuickGELU
        tokens = self.linear_2(tokens)
        tokens = tokens + residue

        return tokens

class CLIP(nn.Module):
    """ Main CLIP text embedder. """
    def __init__(self):
        super().__init__()

        # vocab size = 49408
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.to(torch.long)
        embeddings = self.embedding(tokens)

        for layer in self.layers:
            embeddings = layer(embeddings)
        
        embeddings = self.layernorm(embeddings)

        return embeddings