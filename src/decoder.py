import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    """ VAE residual block with residual connection for learning local features and increasing channels. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels % 32 == 0, "in_channels must be divisible by 32"
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        assert out_channels % 32 == 0, "out_channels must be divisible by 32"
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels,  kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image/image latent: (B, C, H, W)

        Returns: processed image/image latent: (B, C', H, W)
        """
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x += self.residual_layer(residue)

        return x 
    
class VAE_AttentionBlock(nn.Module):
    """ VAE attention block with residual connection for learning global image features. """
    def __init__(self, channels):
        super().__init__()
        assert channels % 32 == 0, "channels must be divisible by 32"
        self.groupnorm_1 = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image/image latent: (B, C, H, W)

        Returns: self-attended image/image latent: (B, C, H, W)
        """
        residue = x

        x = self.groupnorm_1(x)
        
        # prep image shape for attention
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = x.transpose(1, 2)

        x = self.attention(x)

        x = x.transpose(1, 2)
        x = x.reshape(b, c, h, w)

        x += residue

        return x

class VAE_Decoder(nn.Sequential):
    """ VAE decoder for dimension expansion. """
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # upsample (interpolate) + conv instead of transposed convolution 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # project back to 3 (RGB) channels
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, 4, 64, 64)

        Returns: image: (B, 3, 512, 512)
        """
        # undo constant scaling
        x = x / 0.18215
        
        for module in self:
            x = module(x)
        
        return x