import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):
    """ Main diffusion class. """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: latent image: (B, 4, 64, 64)
            context: CLIP text embedding of prompt: (B, 77, 768)
            time: sinusoidal time embedding (1, 320) - batch is 1
        
        Returns: predicted noise latent: (B, 4, 64, 64)
        """
        time = self.time_embedding(time)

        latent = self.unet(latent, context, time)

        output = self.final(latent)

        return output

class UNET(nn.Module):
    """ UNET for denoising image latent. """
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([

            # x: (B, 4, 64, 64)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # x: (B, 320, 64, 64)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # x: (B, 640, 32, 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # x: (B, 1280, 16, 16)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # x: (B, 1280, 8, 8)
        ])

        # x: (B, 1280, 8, 8)
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )
        # x: (B, 1280, 8, 8)

        self.decoders = nn.ModuleList([
            # x: (B, 1280, 8, 8)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

            # x: (B, 1280, 16, 16)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            
            # x: (B, 1280, 32, 32)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            # x: (B, 640, 64, 64)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            # final UpSample (projection) layer to 4 channels is outside the UNET
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, 4, 64, 64)
            context: prompt embedding: (B, 77, 768)
            time: time embedding: (1, 1280) - 1 is batch

        Returns: processed latent: (B, 320, 64, 64)
        """

        # encoder
        skips = []
        for encoder in self.encoders:
            x = encoder(x, context, time)
            skips.append(x)
        
        # bottleneck
        x = self.bottleneck(x, context, time)

        # decoder
        for decoder in self.decoders:
            skip = skips[-1]
            skips.pop(-1)
            x = decoder(torch.cat((x, skip), dim=1), context, time)

        return x
    
class UNET_AttentionBlock(nn.Module):
    """ UNET attention block. """
    def __init__(self, n_head, d_k, d_context=768):
        super().__init__()
        d_model = n_head * d_k

        assert d_model % 32 == 0, "channels must be divisible by 32"
        self.groupnorm = nn.GroupNorm(32, d_model, eps=1e-6)
        self.conv_input = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.self_attention = SelfAttention(n_head, d_model, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(d_model)

        self.cross_attention = CrossAttention(n_head, d_model, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, 4 * d_model * 2)
        self.linear_geglu_2 = nn.Linear(4 * d_model, d_model)

        self.conv_output = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, C, H, W)
            context: prompt embedding: (B, 77, 768)

        Returns: self and cross-attended latent: (B, C, H, W)
        """
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        # self-attention with skip connection
        # (B, C, H, W) -> (B, C, seq_len) -> (B, seq_len, C)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        residue_short = x
        x = self.layernorm_1(x)
        x = self.self_attention(x)
        x = x + residue_short

        # cross-attention with skip connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.cross_attention(x, context)
        x = x + residue_short

        # feedforward with GeGLU and skip connection
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + residue_short

        # output conv
        # (B,seq_len,C) -> (B,C,H,W)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.conv_output(x)

        x = x + residue_long

        return x      

class UNET_ResidualBlock(nn.Module):
    """ UNET residual block. """
    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()

        assert in_channels % 32 == 0, "in_channels must be divisible by 32"
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(d_time, out_channels)

        assert out_channels % 32 == 0, "out_channels must be divisible by 32"
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image latent: (B, C, H, W)
            time: time embedding: (1, 1280)

        Returns: processed latent: (B, C', H, W)
        """        
        residue = self.residual_layer(x)

        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)

        time = F.silu(time)
        time = self.linear_time(time)

        # broadcast and add x and time
        time = time.unsqueeze(-1).unsqueeze(-1)
        # time acts as a channel-wise bias
        merged = x + time

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        out = merged + residue

        return out

class UNET_OutputLayer(nn.Module):
    """ Final UNET layer that projects back to 4 latent channels. """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x

class TimeEmbedding(nn.Module):
    """ Simple time embedding MLP. """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: sinusoidal time embedding: (1, 320)

        Returns: learned time embedding: (1, 1280)
        """
        time = self.linear_1(time)
        time = F.silu(time)
        time = self.linear_2(time)

        return time

class SwitchSequential(nn.Sequential):
    """ Helper class to chain Modules with conditionals. """
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UpSample(nn.Module):
    """ Helper class to upsample latents in the UNET. """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W) -> (B,C,2*H,2*W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x