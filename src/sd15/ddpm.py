import torch


class DDPM:
    def __init__(
        self,
        n_step_train=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):

        self.betas = (
            torch.linspace(
                start=beta_start**0.5, end=beta_end**0.5, steps=n_step_train
            )
            ** 2
        )

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas, dim=0
        ) 

    def q_sample(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        generator: torch.Generator
    ) -> torch.Tensor:
        """Adds Gaussian noise to latent image at given timestep using
        formula 4 of the DDPM paper.

        Args:
            latents: (B, 4, 64, 64)
            timesteps: (B,)
        
        Returns:
            (B, 4, 64, 64)
        """

        alpha_bar_ts = (self.alpha_bars[timesteps]).to(
            device=latents.device, dtype=latents.dtype
        )
        alpha_bar_ts = alpha_bar_ts.flatten()

        # (B,) -> (B, 1, 1, 1)
        while len(alpha_bar_ts.shape) < len(latents.shape):
            alpha_bar_ts = alpha_bar_ts.unsqueeze(-1)

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )

        return (alpha_bar_ts ** 0.5) * latents + ((1 - alpha_bar_ts) ** 0.5) * noise

    def p_sample(
        self,
        latent: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator
    ) -> torch.Tensor:
        """
        Samples the previous latent using predicted noise at timestep t via
        formula 7 and 15 of DDPM paper.

        Args:
            latent: (B, 4, 64, 64)
            noise_pred: (B, 4, 64, 64)
            t: current timestep
            t_prev: previous timestep
        
        Returns:
            (B, 4, 64, 64)
        """
        
        alpha_bar_t = (self.alpha_bars[t]).to(device=latent.device, dtype=latent.dtype)
        if t_prev >= 0:
            alpha_bar_t_prev = (self.alpha_bars[t_prev]).to(
                device=latent.device, dtype=latent.dtype
            )
        else:
            alpha_bar_t_prev = 1

        # alpha_t = alpha_bar_t / alpha_bar_t_prev because inference does not take unit denoising steps
        alpha_t = (alpha_bar_t / alpha_bar_t_prev).to(
            device=latent.device, dtype=latent.dtype
        )

        # approximate x0 via formula 15
        x0_pred = (latent - ((1 - alpha_bar_t) ** 0.5) * noise_pred) / (
            alpha_bar_t**0.5
        )

        # sampling
        coeff_x0 = (alpha_bar_t_prev**0.5) * (1 - alpha_t) / (1 - alpha_bar_t)
        coeff_xt = (alpha_t**0.5) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean = coeff_x0 * x0_pred + coeff_xt * latent

        # final denoising step is not stochastic
        if t_prev < 0:
            return mean

        noise = torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        )

        variance = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * (1 - alpha_t)
        variability = noise * ((variance.clamp(min=1e-20)) ** 0.5)

        return mean + variability