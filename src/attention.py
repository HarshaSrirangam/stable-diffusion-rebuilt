import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi-head self attention for learning global image features and for CLIP.
    
    Args:
        n_head: number of heads
        d_model: embedding dimension
        in_proj_bias: whether or not to include bias term for QKV linear layer
        out_proj_bias: whether or not to include bias term for final projection
    """
    def __init__(self, n_head, d_model, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # one big W for QKV
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=in_proj_bias) 
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias) 
        self.n_head = n_head
        assert d_model% n_head == 0, "embed_dim must be divisible by n_head"
        self.d_k = d_model // n_head

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        """
        Args:
            x: 
                Image features: (B, H*W, d_model)
                CLIP embeddings: (B, 77, d_model)
            causal_mask: whether or not tokens should attend to future tokens

        Returns: (B, seq_len, d_model) self-attended features
        """
        latent_shape = x.shape
        b, seq_len, _ = latent_shape
        interim_shape = (b, seq_len, self.n_head, self.d_k)

        x = self.in_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=-1)

        # split into heads: x: (B,seq_len,d_model) -> (B,seq_len,n_head,d_k) -> (B,n_head,seq_len,d_k)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        attn_scores = q @ k.transpose(-1, -2)

        # causal mask: a token shouldn't attend to tokens after it (used for CLIP)
        # triu because masked_fill operates on true values
        # triu(1) because diagonal itself should stay False (so invert makes it True)
        # implement by setting attn score to -inf so softmax = 0
        if causal_mask:
            mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # normalize by d_k and softmax
        attn_scores /= (self.d_k ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        x = attn_scores @ v

        # merge heads: (B,n_head,seq_len, d_k) -> (B, seq_len, n_head, d_k) -> (B, seq_len, d_model)
        x = x.transpose(1, 2)
        x = x.reshape(b, seq_len, -1)

        out = self.out_proj(x)

        return out
    
class CrossAttention(nn.Module):
    """
    Multi-head cross attention for conditioning image features on context.
    Queries from image, keys and values from context.
    
    Args:
        n_head: number of heads
        d_model: embedding dimension
        d_context: CLIP embedding dimension
        in_proj_bias: whether or not to include bias term for QKV linear layers
        out_proj_bias: whether or not to include bias term for final projection
    """
    def __init__(self, n_head, d_model, d_context, in_proj_bias=True, out_proj_bias=True):
        super().__init__(self)
        self.q_proj = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, d_model, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_head = n_head
        assert d_model % n_head == 0, "embed_dim must be divisible by n_head"
        self.d_k = d_model // n_head

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features: (B, H*W, d_model)
            context: CLIP embeddings: (B, 77, d_context)

        Returns: context-attended image features: (B, H*W, d_model)
        """
        latent_shape = x.shape
        b, _, _ = latent_shape
        # -1 for seq so it is inferred based on x vs context
        interim_shape = (b, -1, self.n_head, self.d_k)
        
        q = self.q_proj(x) # (B,H*W,d_model)
        k = self.k_proj(context) # (B,77,d_model)
        v = self.v_proj(context) # (B,77,d_model)

        # split into heads: (B,seq_len,d_model) -> (B,seq_len,n_head,d_k) -> (B,n_head,seq_len,d_k)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B,n_head,H*W,d_k) @ (B,n_head,d_k,77) -> (B,n_head,H*W,77)
        attn_scores = q @ k.transpose(-1, -2)
        attn_scores /= (self.d_k ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # (B,n_head,H*W,77) @ (B,n_head,77,d_k) -> (B,n_head,H*W,d_k)
        x = attn_scores @ v

        # merge heads
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(latent_shape)

        out = self.out_proj(x)

        return out