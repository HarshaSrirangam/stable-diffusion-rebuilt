import torch
import torch.nn as nn
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, seq_length):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(seq_length, embed_dim))

    def forward(self, tokens: torch.Tensor):
        # tokens: (B,77) -> (B,77,768)
        assert tokens.shape[1] == 77, "only a max of 77 tokens allowed"
        return self.token_embedding_table(tokens) + self.position_embeddings
    
class CLIPLayer(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.attention = SelfAttention(n_heads, embed_dim)
        self.linear_1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear_2 = nn.Linear(4*embed_dim, embed_dim)
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.FloatTensor) -> torch.FloatTensor:
        # tokens: (B,77,768) -> (B,77,768) 
        
        assert tokens.shape[1:] == (77, 768), "embed_dim must be 768"
        # multi-head attention (with residual connections and layernorm)
        residue = tokens
        tokens = self.layernorm_1(tokens)
        tokens = self.attention(tokens, causal_mask=True)
        tokens = tokens + residue

        # feed-forward (with residual connections and layernorm)
        residue = tokens
        tokens = self.layernorm_2(tokens)
        tokens = self.linear_1(tokens)
        tokens = tokens * torch.sigmoid(1.702 * tokens) # QuickGELU
        tokens = self.linear_2(tokens)
        tokens = tokens + residue

        return tokens

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # vocab size, embed_dim, max seq_length
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            # heads, embed_dim
            CLIPLayer(12, 768) for _ in range(12)
        ])

        # per-token normalization (over is 768 features)
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.to(torch.long)

        embeddings = self.embedding(tokens)
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        embeddings = self.layernorm(embeddings)

        return embeddings
 