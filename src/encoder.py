import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    """ VAE encoder for dimension reduction. """
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # 512 -> 8 latent channels (first 4 channels are 4 means, second 4 channels are 4 log variances)
            # for each of the 4 latent variables
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image: (B, 3, 512, 512)
            noise: Gaussian noise sampled from standard normal: (B, 4, 64, 64)

        Returns: image latent: (B, 4, 64, 64)
        """
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # pad right and bottom by 1 because stride of 2 is messy
                # (B,C,H,W) -> (B,C,H+1,W+1)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # clamp variances
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = torch.exp(log_variance)
        stdev = torch.sqrt(variance)

        # sample from this distribution
        # X ~ N(mu, sigma^2), Z ~ N(0, 1) -> X ~ mu + sigma * Z
        x = mean + stdev * noise

        # scale by constant
        x = x * 0.18215

        return x