import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):
    """Helper class to upsample latents in the UNET."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W) -> (B,C,2*H,2*W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        return x


class TimeEmbedding(nn.Module):
    """Time embedding with sinusoidal features and simple MLP."""

    def __init__(self, d_embed=320):
        super().__init__()
        half_d_embed = d_embed // 2
        freqs = torch.pow(
            10000,
            -torch.arange(start=0, end=half_d_embed, dtype=torch.float32)
            / half_d_embed,
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.linear1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,)

        Returns: (B, 1280)
        """
        timesteps = timesteps.to(dtype=torch.float32)
        freqs = self.freqs.to(device=timesteps.device, dtype=torch.float32)

        timesteps = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
        timesteps = torch.cat((torch.cos(timesteps), torch.sin(timesteps)), dim=-1)

        timesteps = self.linear1(timesteps)
        timesteps = F.silu(timesteps)
        timesteps = self.linear2(timesteps)

        return timesteps


class TransformerBlock(nn.Module):
    """UNet transformer block with SDPA."""

    def __init__(self, n_head, channels, d_context=768):
        super().__init__()

        self.n_head = n_head
        assert channels % n_head == 0, "channels must be divisible by n_head"
        self.d_k = channels // n_head

        assert channels % 32 == 0, "channels must be divisible by 32"
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)

        self.q1 = nn.Linear(channels, channels, bias=False)
        self.k1 = nn.Linear(channels, channels, bias=False)
        self.v1 = nn.Linear(channels, channels, bias=False)
        self.proj_out1 = nn.Linear(channels, channels)

        self.layernorm2 = nn.LayerNorm(channels)

        self.q2 = nn.Linear(channels, channels, bias=False)
        self.k2 = nn.Linear(d_context, channels, bias=False)
        self.v2 = nn.Linear(d_context, channels, bias=False)
        self.proj_out2 = nn.Linear(channels, channels)

        self.layernorm3 = nn.LayerNorm(channels)

        self.linear_geglu1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu2 = nn.Linear(4 * channels, channels)

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        residual0 = x
        x = self.groupnorm(x)
        x = self.proj_in(x)

        b, c, h, w = x.shape

        # 1) SELF ATTENTION
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.reshape(b, c, -1).transpose(-1, -2)
        residual1 = x
        x = self.layernorm1(x)

        # QKV projections and split into heads
        q1 = self.q1(x).reshape(b, h * w, self.n_head, self.d_k).transpose(1, 2)
        k1 = self.k1(x).reshape(b, h * w, self.n_head, self.d_k).transpose(1, 2)
        v1 = self.v1(x).reshape(b, h * w, self.n_head, self.d_k).transpose(1, 2)

        x = F.scaled_dot_product_attention(q1, k1, v1) # (B, n_heads, H*W, d_k)

        # merge heads
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(b, h * w, c)

        x = self.proj_out1(x)
        x = x + residual1

        # 2) CROSS ATTENTION
        residual1 = x
        x = self.layernorm2(x)

        # QKV projections
        q2 = self.q2(x)
        k2 = self.k2(context)
        v2 = self.v2(context)

        b, seq_len_x, c = q2.shape
        _, seq_len_context, _ = k2.shape

        # split into heads
        q2 = q2.reshape(b, seq_len_x, self.n_head, self.d_k).transpose(1, 2)
        k2 = k2.reshape(b, seq_len_context, self.n_head, self.d_k).transpose(1, 2)
        v2 = v2.reshape(b, seq_len_context, self.n_head, self.d_k).transpose(1, 2)

        x = F.scaled_dot_product_attention(q2, k2, v2)

        # merge heads
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(b, seq_len_x, c)

        x = self.proj_out2(x)
        x = x + residual1

        residual1 = x
        x = self.layernorm3(x)

        # 3) GeGLU FEED FORWARD
        x, gate = self.linear_geglu1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu2(x)

        x = x + residual1

        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x = x.transpose(-1, -2).contiguous()
        x = x.reshape(b, c, h, w)

        x = self.proj_out(x)
        x = x + residual0

        return x


class ResBlock(nn.Module):
    """UNET residual block."""

    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

        assert in_channels % 32 == 0, "in_channels must be divisible by 32"
        self.groupnorm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(d_time, out_channels)

        assert out_channels % 32 == 0, "out_channels must be divisible by 32"
        self.groupnorm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, C, H, W)
            time: time embedding: (B or 1, 1280)

        Returns:
            processed latent: (B, C', H, W)
        """
        residual = self.shortcut(x)

        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time = F.silu(time)
        time = self.time(time)

        # broadcast and add x and time element-wise
        time = time.unsqueeze(-1).unsqueeze(-1)
        # time acts as a channel-wise bias
        out = x + time

        out = self.groupnorm2(out)
        out = F.silu(out)
        out = self.conv2(out)

        return out + residual


class UNetBlock(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, TransformerBlock):
                x = module(x, context)
            elif isinstance(module, ResBlock):
                x = module(x, time)
            else:
                x = module(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = TimeEmbedding()

        self.down_blocks = nn.ModuleList(
            [
                # x: (B, 4, 64, 64)
                UNetBlock(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                UNetBlock(ResBlock(320, 320), TransformerBlock(8, 320)),
                UNetBlock(ResBlock(320, 320), TransformerBlock(8, 320)),
                # x: (B, 320, 64, 64)
                UNetBlock(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
                UNetBlock(ResBlock(320, 640), TransformerBlock(8, 640)),
                UNetBlock(ResBlock(640, 640), TransformerBlock(8, 640)),
                # x: (B, 640, 32, 32)
                UNetBlock(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
                UNetBlock(ResBlock(640, 1280), TransformerBlock(8, 1280)),
                UNetBlock(ResBlock(1280, 1280), TransformerBlock(8, 1280)),
                # x: (B, 1280, 16, 16)
                UNetBlock(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
                UNetBlock(ResBlock(1280, 1280)),
                UNetBlock(ResBlock(1280, 1280)),
                # x: (B, 1280, 8, 8)
            ]
        )

        # x: (B, 1280, 8, 8)
        self.bottleneck_blocks = UNetBlock(
            ResBlock(1280, 1280), TransformerBlock(8, 1280), ResBlock(1280, 1280)
        )
        # x: (B, 1280, 8, 8)

        self.up_blocks = nn.ModuleList(
            [
                # x: (B, 1280, 8, 8)
                UNetBlock(ResBlock(2560, 1280)),
                UNetBlock(ResBlock(2560, 1280)),
                UNetBlock(ResBlock(2560, 1280), UpSample(1280)),
                # x: (B, 1280, 16, 16)
                UNetBlock(ResBlock(2560, 1280), TransformerBlock(8, 1280)),
                UNetBlock(ResBlock(2560, 1280), TransformerBlock(8, 1280)),
                UNetBlock(
                    ResBlock(1920, 1280), TransformerBlock(8, 1280), UpSample(1280)
                ),
                # x: (B, 1280, 32, 32)
                UNetBlock(ResBlock(1920, 640), TransformerBlock(8, 640)),
                UNetBlock(ResBlock(1280, 640), TransformerBlock(8, 640)),
                UNetBlock(ResBlock(960, 640), TransformerBlock(8, 640), UpSample(640)),
                # x: (B, 640, 64, 64)
                UNetBlock(ResBlock(960, 320), TransformerBlock(8, 320)),
                UNetBlock(ResBlock(640, 320), TransformerBlock(8, 320)),
                UNetBlock(ResBlock(640, 320), TransformerBlock(8, 320)),
            ]
        )

        # final UpSample (projection) layer to 4 channels
        self.proj_final = nn.ModuleList(
            [
                nn.GroupNorm(32, 320, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(320, 4, kernel_size=3, padding=1),
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, 4, 64, 64)
            context: prompt embedding: (B, 77, 768)
            time: (B,)

        Returns: processed latent: (B, 4, 64, 64)
        """
        # time embedding
        time = self.time_emb(time)

        # down
        skips = []
        for down in self.down_blocks:
            x = down(x, context, time)
            skips.append(x)

        # bottleneck
        x = self.bottleneck_blocks(x, context, time)

        # up
        for up in self.up_blocks:
            skip = skips[-1]
            skips.pop(-1)
            x = up(torch.cat((x, skip), dim=1), context, time)

        # final projection
        for module in self.proj_final:
            x = module(x)

        return x
