import torch


class DDPM:
    def __init__(
        self,
        n_step_train=1000,
        n_step_inf=50,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        if n_step_train <= 0:
            raise ValueError("n_step_train must be positive")
        if n_step_inf <= 0:
            raise ValueError("n_step_inf must be positive")
        if n_step_train % n_step_inf != 0:
            raise ValueError("n_step_train must be divisible by n_step_inf")

        self.n_step_train = n_step_train
        self.n_step_inf = n_step_inf

        # DDPM params
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

        # timesteps
        step_size = n_step_train // n_step_inf
        self.full_timesteps = torch.arange(
            start=n_step_train - 1,
            end=-1,
            step=-step_size,
            dtype=torch.long
        )
        # one sampler per inference method call
        # so timesteps can be indexed directly since its account for n_step_inf
        self.timesteps = self.full_timesteps

    def set_strength(self, strength: float) -> None:
        """Modifies self.timesteps accordingly"""
        if not 0.0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        # [999, 979, 959, ... 19] -> [979, 959, ... 19]
        num_steps = int(strength * len(self.full_timesteps))
        start_idx = len(self.full_timesteps) - num_steps
        self.timesteps = self.full_timesteps[start_idx:] 

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        """Adds Gaussian noise to latent at given timestep using
        formula 4 of the DDPM paper.

        Args:
            latents: (B, 4, 64, 64)
            noise: (B, 4,  64, 64)
            timesteps: (B,)
        
        Returns:
            (B, 4, 64, 64)
        """
        if timesteps is None:
            # assumes we set strength prior to this, meaning strength was 0 and we just return latent
            if len(self.timesteps) == 0:
                return latents
            # turn all samples in batch to pure noise
            # turn timesteps into tensor of shape (B,) 
            timesteps = torch.full( # why not just use full like to be consistent with step()?
                size=(latents.shape[0],), # size must be a tuple
                fill_value=int(self.timesteps[0].item()), # not necessarily pure noise, just max noise defined by self.timeteps
                device=latents.device,
                dtype=torch.long
            )
        else:
            timesteps = torch.tensor(timesteps, device=latents.device, dtype=torch.long) # incase input isnt a tensor
            if timesteps.ndim == 0: # turn into shape (B,)
                timesteps = timesteps.expand(latents.shape[0])

        # At this point, timesteps is a tensor of shape (B,), each entry being one latent's corresponding timestep to add noise to

        # alpha_bar_ts becomes a tensor of shape (B,) since its indexed by timesteps
        alpha_bar_ts = (self.alpha_bars[timesteps]).to(
            device=latents.device, dtype=latents.dtype
        )
        #alpha_bar_ts = alpha_bar_ts.flatten()

        # (B,) -> (B, 1, 1, 1)
        while len(alpha_bar_ts.shape) < len(latents.shape):
            alpha_bar_ts = alpha_bar_ts.unsqueeze(-1)

        # returns noisified latets of shape (B, 4, 64, 64)
        return (alpha_bar_ts ** 0.5) * latents + ((1 - alpha_bar_ts) ** 0.5) * noise
    
    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor | int, # bad # timestep is shared
        prev_timestep: torch.Tensor | int | None, # bad
        latents: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Samples the previous latent using predicted noise at timestep t via
        formula 7 and 15 of DDPM paper.

        Args:
            latent: (B, 4, 64, 64)
            noise_pred: (B, 4, 64, 64)
            timestep: current timestep
            prev_timestep: previous timestep
        
        Returns:
            (B, 4, 64, 64)
        """
        t = torch.tensor(timestep, device=latents.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(latents.shape[0]) # int -> (B,)

        if prev_timestep is None:
            t_prev = torch.full_like(t, -1)
        else:
            t_prev = torch.tensor(prev_timestep, device=latents.device, dtype=torch.long)
            if t_prev.ndim == 0:
                t_prev = t_prev.expand(latents.shape[0]) # int -> (B,)

        # By now, t and t_prev are shape (B,)

        # (B,)
        alpha_bar_t = self.alpha_bars[t].to(device=latents.device, dtype=latents.dtype)
        # alpha_bar_t_prev is 1 for final timesteps, so default to 1
        alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

        valid_prev = t_prev >= 0
        if valid_prev.any():
            alpha_bar_t_prev[valid_prev] = self.alpha_bars[t_prev[valid_prev]].to(
                device=latents.device,
                dtype=latents.dtype,
            )

        # by now, alpha_bar_t_prev is:
            # shape (B,)
            # looks something like [some alpha, some alpha, 1, some alpha, 1, 1 ...]
            # assuming prev_timesteps arg looks like [valid prev, valid prev, None or negative number???]

        while len(alpha_bar_t.shape) < len(latents.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)

        # by now alpha_bar_t and alpha_bar_t prev are (B, 1, 1, 1), now alpha_t is as well
        alpha_t = alpha_bar_t / alpha_bar_t_prev

        x0_pred = (latents - ((1.0 - alpha_bar_t) ** 0.5) * noise_pred) / (alpha_bar_t**0.5)

        coeff_x0 = (alpha_bar_t_prev**0.5) * (1.0 - alpha_t) / (1.0 - alpha_bar_t)
        coeff_xt = (alpha_t**0.5) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
        mean = coeff_x0 * x0_pred + coeff_xt * latents

        # by now mean is of shape (B, 4, 64, 64)
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # noise is of shape (B, 4, 64, 64)

        variance = ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_t)
        # variance is of shape (B, 1, 1, 1)
        sample = mean + noise * (variance.clamp(min=1e-20) ** 0.5)

        final_mask = (t_prev < 0)
        while final_mask.ndim < latents.ndim:
            final_mask = final_mask.unsqueeze(-1)

        # final_mask: (B, 1, 1, 1)
        # if it was last timestep return only mean no stochasticity/sampling
        return torch.where(final_mask, mean, sample)