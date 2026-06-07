import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class ResBlock(nn.Module):
    """VAE residual block for learning local features and increasing channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels % 32 == 0, "in_channels must be divisible by 32"
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        assert out_channels % 32 == 0, "out_channels must be divisible by 32"
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: image/image latent: (B, C, H, W)
        """
        residual = x
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        x += self.shortcut(residual)

        return x


class AttnBlock(nn.Module):
    """VAE single-head attention block with residual connection for learning global image features."""

    def __init__(self, channels):
        super().__init__()
        assert channels % 32 == 0, "channels must be divisible by 32"
        self.groupnorm1 = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.d_k = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.groupnorm1(x)

        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        q = q.reshape(b, c, -1).transpose(-1, -2)
        k = k.reshape(b, c, -1).transpose(-1, -2)
        v = v.reshape(b, c, -1).transpose(-1, -2)

        attn_scores = q @ k.transpose(-1, -2)
        attn_scores = F.softmax(attn_scores / (self.d_k**0.5), dim=-1)

        x = attn_scores @ v

        # (B, H*W, C)-> (B, C, H*W) -> (B, C, H, W)
        x = x.transpose(-1, -2).reshape(b, c, h, w)
        x = self.proj_out(x)

        x += residual

        return x


class Encoder(nn.Module):
    """VAE encoder for dimension reduction."""

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResBlock(128, 128),
            ResBlock(128, 128),
            DownSample(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            DownSample(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            DownSample(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            AttnBlock(512),
            ResBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            # 512 -> 8 latent channels (first 4 channels are 4 means, second 4 channels are 4 log variances)
            # for each of the 4 latent variables
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.blocks:
            x = module(x)

        return x


class Decoder(nn.Module):
    """VAE decoder for dimension expansion."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResBlock(512, 512),
            AttnBlock(512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            # upsample (interpolate) + conv instead of transposed convolution
            UpSample(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            UpSample(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            UpSample(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # project back to 3 (RGB) channels
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.blocks:
            x = module(x)

        return x


class AutoEncoder(nn.Module):
    """VAE for image encoding/decoding"""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)

    def encode(self, images: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Encodes images to 4 latent features.
        Args:
            image: (B, 3, 512, 512)
            noise: (B, 4, 64, 64)

        Returns:
            (B, 4, 64, 64)
        """
        noise = noise.to(device=images.device, dtype=images.dtype)
        latents = self.encoder(images)
        latents = self.quant_conv(latents)
        mean, log_variance = torch.chunk(latents, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = torch.exp(log_variance)
        stdev = torch.sqrt(variance)

        if noise is None:
            return 0.18215 * mean

        return 0.18215 * (mean + stdev * noise)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes image back to RGB.
        Args:
            latents: (B, 4, 64, 64)

        Returns:
            (B, 3, 512, 512)
        """
        latents = latents / 0.18215
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def forward(self, images: torch.Tensor, noise: torch.Tensor = None):
        """Encodes and decodes image.
        Args:
            image: (B, 3, 512, 512)
            noise: (B, 4, 64, 64)

        Returns:
            (B, 3, 512, 512)
        """
        noise = noise.to(device=images.device, dtype=images.dtype)
        latents = self.encode(images, noise)
        images = self.decode(latents)

        return images
