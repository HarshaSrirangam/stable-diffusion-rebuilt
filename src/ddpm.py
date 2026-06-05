import torch
import numpy as np
import math

class DDPMSampler:
    def __init__(
        self, generator: torch.Generator, 
        num_training_steps=1000, 
        beta_start: float = 0.00085, # beta_0
        beta_end: float = 0.0120 # beta_999
    ):
        # earlier steps: low noise, later steps: high noise
        self.betas = torch.linspace(start=beta_start**0.5, end=beta_end**0.5, steps=num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.arange(start=num_training_steps - 1, end=-1, step=-1)

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        
        step_len = self.num_training_steps // self.num_inference_steps
        self.timesteps = torch.arange(start=self.num_training_steps - step_len, end=-1, step=-step_len)

    def add_noise(self, latents: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        alpha_bars = self.alpha_bars.to(device=latents.device, dtype=latents.dtype)
        timesteps = timesteps.to(latents.device)

        # alpha_bar
        sqrt_alpha_bars = alpha_bars[timesteps] ** 0.5
        sqrt_alpha_bars = sqrt_alpha_bars.flatten()
        while len(sqrt_alpha_bars.shape) < len(latents.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)

        # 1 - alpha_bar
        sqrt_one_minus_alpha_bars = (1 - alpha_bars[timesteps]) ** 0.5
        sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.flatten()
        while len(sqrt_one_minus_alpha_bars) < len(latents.shape):
            sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.unsqueeze(-1)

        # add noise: latent' = sqrt(alpha_bar) * latent + sqrt(1-alpha_bar) * noise
        epsilon = torch.randn(latents.shape, generator=self.generator, device=latents.device, dtype=latents.dtype)
        noisy_latents = sqrt_alpha_bars * latents + sqrt_one_minus_alpha_bars * epsilon

        return noisy_latents

    def step(self, timestep: int, latents: torch.Tensor, model_outputs: torch.Tensor) -> torch.Tensor:
        t = timestep
        t_prev = self._get_prev_timestep(t)
        one = self.one.to(latents.device, latents.dtype)

        alpha_bar_t = (self.alpha_bars[t]).to(latents.device, latents.dtype)
        alpha_bar_t_prev = (self.alpha_bars[t_prev] if t_prev >= 0 else one).to(latents.device, latents.dtype)

        alpha_t_jump = (alpha_bar_t / alpha_bar_t_prev).to(latents.device, latents.dtype)

        # compute predicted x0 using forumla 15 of DDPM paper
        pred_x0 = (latents - ((1 - alpha_bar_t) ** 0.5) * model_outputs) / (alpha_bar_t ** 0.5)

        # using above, mean of reverse noise distribution (formula 7)
        x0_coeff = (alpha_bar_t_prev ** 0.5) * (1 - alpha_t_jump) / (1 - alpha_bar_t)
        xt_coeff = (alpha_t_jump ** 0.5) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean = x0_coeff * pred_x0 + xt_coeff * latents

        # compute variance/std for reverse noise distribution (formula 7)
        variance = 0
        if t_prev >= 0:
            # this means current timestep is not the first, so variance is nonzero
            noise = torch.randn(latents.shape, generator=self.generator, device=latents.device, dtype=latents.dtype)
            variance = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * (1 - alpha_t_jump)
            variance = noise * (variance.clamp(min=1e-20) ** 0.5)

        pred_x_prev = mean + variance

        return pred_x_prev
    
    def set_strength(self, strength=1):
        if 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]  
    
    def _get_prev_timestep(self, timestep: int):
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)

        return prev_t
