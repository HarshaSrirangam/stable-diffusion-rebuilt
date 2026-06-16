"""Oiler sampler"""
import torch

class Euler:
    def __init__(self):
        self.init_noise_scale = torch.tensor(1.0)

    def scale_model_input(latents, timestep):
        pass

    def add_noise(self, latents, noise, timesteps):
        pass

    def step(
        noise_pred,
        timestep,
        prev_timestep,
        latents,
        generator
    ):
        pass