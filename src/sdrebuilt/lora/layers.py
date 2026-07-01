import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False # freeze pretrained weights
        in_f, out_f = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn((r, in_f), dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros((out_f, r), dtype=torch.float32))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor):
        # x: (batch, in_f)
        base = self.base_layer(x)

        # (x @ A.T): (batch, in_f) @ (in_f, rank) -> (batch, rank)
        bottleneck = x @ self.A.T
        # (x @ B.T): (batch, rank) @ (rank, out_f) -> (batch, out_f)
        correction = bottleneck @ self.B.T

        return base + self.scaling(correction)



