import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # x: (B,C,H,W)
            # 3 -> 128 features
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # maintain
            VAE_ResidualBlock(128, 128),
            # maintain
            VAE_ResidualBlock(128, 128),
            # ~ H/2, W/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # 128 -> 256 features
            VAE_ResidualBlock(128, 256),
            # maintain
            VAE_ResidualBlock(256, 256),
            # ~ H/2, W/2
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # 256 -> 512 features
            VAE_ResidualBlock(256, 512),
            # maintain
            VAE_ResidualBlock(512, 512),
            # ~ H/2, W/2
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
              
            # maintain
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # SHAPE: (B,512,H/8,W/8)
            # Convs capture local features, Attention captures global features
            # maintain
            VAE_AttentionBlock(512),

            # maintain
            VAE_ResidualBlock(512, 512),

            # split channels into groups of 16, normalize pixel values within group (and apply learned weight and bias)
            nn.GroupNorm(32, 512),

            # x * sigmoid(x)
            nn.SiLU(),

            # 512 -> 8 features (first 4 features are 4 means, second 4 features are 4 log variances)
            # why log variance? bc activations are arbitrary reals, but variance is strictly positive
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # final projection
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (B,3,512,512)
        # noise: (B,4,64,64)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # pad right and bottom by 1 (bc stride=2 returns odd dim)
                # (B,C,H,W) -> (B,C,H+1,W+1)
                x = F.pad(x, (0, 1, 0, 1))
            # apply layer/module AFTER pre-adjusting size for stride=2
            x = module(x)
        # get mean and logvar
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # so that exp doesn't create very large or small variances
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = torch.exp(log_variance)
        stdev = torch.sqrt(variance)

        # sample from this distribution
        # X ~ N(mu, sigma), Z ~ N(0, 1) -> X ~ mu + sigma * Z
        x = mean + stdev * noise

        # scale by constant
        x = x * 0.18215

        return x




 