import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Linear LoRA layer. Based on:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False # freeze pretrained weights
        d_in, d_out = base_layer.in_features, base_layer.out_features
        device = base_layer.weight.device
        self.A = nn.Parameter(torch.randn((r, d_in), dtype=torch.float32, device=device))
        self.B = nn.Parameter(torch.zeros((d_out, r), dtype=torch.float32, device=device))
        self.scaling = alpha / r
        self.enabled = True
        self.merged = None

    def forward(self, x: torch.Tensor):
        if not self.enabled:
            return self.base_layer(x)
        
        if self.merged is not None:
            return self.merged(x)
        
        # x: (B, d_in)
        base = self.base_layer(x)

        # (x @ A.T): (B, d_in) @ (d_in, r) -> (B, r)
        bottleneck = x @ self.A.T
        # (x @ B.T): (B, r) @ (r, d_out) -> (B, d_out)
        correction = bottleneck @ self.B.T

        return base + self.scaling * correction

    def merge(self) -> None:
        """Creates self.merged: linear layer mergring B, A, and base layer."""
        has_bias = self.base_layer.bias is not None
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=has_bias
        )
        with torch.no_grad(): # must turn off autograd for in-place ops on grad-requiring tensors
            delta = self.scaling * (self.B @ self.A)
            merged.weight.copy_(self.base_layer.weight + delta)
            if has_bias:
                merged.bias.copy_(self.base_layer.bias)
        merged.requires_grad_(False) # turn off in case more training post-merge
        self.merged = merged.to(
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype
        )