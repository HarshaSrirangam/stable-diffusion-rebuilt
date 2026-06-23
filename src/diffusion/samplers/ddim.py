import torch


class DDIM:
    """
    DDIM sampler. Based on:
    Song et al., "Denoising Diffusion Implicit Models"
    https://arxiv.org/abs/2010.02502
    """

    def __init__(
        self,
        n_step_train=1000,
        n_step_inf=50,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        eta: float = 0.0,
    ):
        if n_step_train <= 0:
            raise ValueError("n_step_train must be positive")
        if n_step_inf <= 0:
            raise ValueError("n_step_inf must be positive")
        if n_step_train % n_step_inf != 0:
            raise ValueError("n_step_train must be divisible by n_step_inf")
        if eta < 0.0:
            raise ValueError("eta cannot be negative")

        self.n_step_train = n_step_train
        self.n_step_inf = n_step_inf

        # DDIM params
        self.eta = eta
        self.betas = (
            torch.linspace(start=beta_start**0.5, end=beta_end**0.5, steps=n_step_train)
            ** 2
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # timesteps
        step_size = n_step_train // n_step_inf
        self.full_timesteps = torch.arange(
            start=n_step_train - 1, end=-1, step=-step_size, dtype=torch.long
        )
        # one sampler per inference method call
        # so timesteps can be indexed directly since it accounts for n_step_inf
        self.timesteps = self.full_timesteps

    def set_timesteps(self, strength: float) -> None:
        """Modifies timesteps according img2img inference strength."""
        if not 0.0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # something like [999, 979, 959, ... 19] -> [979, 959, ... 19]
        num_steps = int(strength * len(self.full_timesteps))
        start_idx = len(self.full_timesteps) - num_steps
        self.timesteps = self.full_timesteps[start_idx:]

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Adds Gaussian noise to latent at given timestep also using
        formula 4 of the DDPM paper. Forward process in DDIM is
        identical to DDPM.

        Args:
            latents: (B, 4, 64, 64)
            noise: (B, 4,  64, 64)
            timesteps: (B,)

        Returns:
            (B, 4, 64, 64)
        """
        if timesteps is None:
            # return latent if strength is 0
            if len(self.timesteps) == 0:
                return latents
            # otherwise pure noise
            # timesteps -> (B,)
            timesteps = torch.full(
                size=(latents.shape[0],),  # (B,)
                fill_value=int(self.timesteps[0].item()),
                device=latents.device,
                dtype=torch.long,
            )
        else:
            timesteps = torch.as_tensor(
                timesteps, device=latents.device, dtype=torch.long
            )
            if timesteps.ndim == 0:
                timesteps = timesteps.expand(latents.shape[0])  # (B,)

        # alpha_bar_ts -> (B,)
        alpha_bars = self.alpha_bars.to(
            device=latents.device, dtype=latents.dtype
        )
        alpha_bar_ts = alpha_bars[timesteps]

        # (B,) -> (B, 1, 1, 1) for broadcasting
        while len(alpha_bar_ts.shape) < len(latents.shape):
            alpha_bar_ts = alpha_bar_ts.unsqueeze(-1)

        return (alpha_bar_ts**0.5) * latents + ((1 - alpha_bar_ts) ** 0.5) * noise

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor | int,
        prev_timestep: torch.Tensor | int | None,
        latents: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Samples the previous latent using predicted noise at timestep t via
        formulas 12 and 16 of DDIM paper. Supports batched inference but each
        sample must be at the same timestep.

        Args:
            latents: (B, 4, 64, 64)
            noise_pred: (B, 4, 64, 64)
            timestep: current timestep
            prev_timestep: previous timestep

        Returns:
            (B, 4, 64, 64)
        """
        t = torch.as_tensor(timestep, device=latents.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(latents.shape[0])  # int -> (B,)

        if prev_timestep is None:
            t_prev = torch.full_like(t, -1)
        else:
            t_prev = torch.as_tensor(
                prev_timestep, device=latents.device, dtype=torch.long
            )
            if t_prev.ndim == 0:
                t_prev = t_prev.expand(latents.shape[0]) # int -> (B,)

        alpha_bars = self.alpha_bars.to(
            device=latents.device, dtype=latents.dtype
        )  # -> (B,)
        alpha_bar_t = alpha_bars[t]
        # alpha_bar_t_prev is 1 for final timesteps, so default to 1
        alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

        valid_prev = t_prev >= 0
        if valid_prev.any():
            alpha_bar_t_prev[valid_prev] = alpha_bars[t_prev[valid_prev]]

        # (B,) -> (B, 1, 1, 1)
        while len(alpha_bar_t.shape) < len(latents.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)

        x0_pred = (latents - ((1.0 - alpha_bar_t) ** 0.5) * noise_pred) / (
            alpha_bar_t**0.5
        )  # same as DDPM
        # variance is 0 by default (eta=0)
        sigma_t = (
            self.eta
            * (((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) ** 0.5)
            * ((1 - (alpha_bar_t / alpha_bar_t_prev)) ** 0.5)
        )

        coeff_x0 = alpha_bar_t_prev**0.5
        coeff_epsilon = (1 - alpha_bar_t_prev - (sigma_t**2)).clamp(min=0) ** 0.5

        if self.eta == 0.0:
            return coeff_x0 * x0_pred + coeff_epsilon * noise_pred

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        return coeff_x0 * x0_pred + coeff_epsilon * noise_pred + sigma_t * noise
