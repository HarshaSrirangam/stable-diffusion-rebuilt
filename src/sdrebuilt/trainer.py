import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model.unet import UNet
from .samplers.ddpm import DDPM


class Trainer:
    """
    Finetune method-agnostic trainer class. Supports UNet finetuning only.

    Args:
        unet: UNet with only intended trainable params active
        dataloader: DataLoader of pre-encoded batched image-caption pairs
        optimizer: Adam or AdamW
        sampler: DDPM sampler
        device: training device
        n_epochs: number of training epochs
        log_interval: number of batches between logging loss
        run_dir: runs/<current_run>
    """

    def __init__(
        self,
        unet: UNet,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        sampler: DDPM,
        device: torch.device,
        n_epochs: int,
        log_interval: int,
        run_dir: Path,
    ):
        self.unet = unet
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.sampler = sampler
        self.device = device
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.run_dir = run_dir

        self.losses = []

    def train(self):
        """
        Trains UNet for n_epochs. Uses bf16 autocast.
        """
        self.unet.train()
        for epoch in range(self.n_epochs):
            pbar = tqdm(
                self.dataloader,
                desc=f"epoch {epoch + 1}/{self.n_epochs}",
                colour="blue",
            )
            for step, (latents, context) in enumerate(pbar):
                latents = latents.to(device=self.device)
                context = context.to(device=self.device)
                b = latents.shape[0]

                # bf16 autocast
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    # sample random timesteps in [0, 999]
                    timesteps = torch.randint(
                        low=0, high=self.sampler.n_step_train, size=(b,)
                    ).to(device=self.device, dtype=torch.long)
                    # sample and add noise to latents
                    sampler_noise = torch.randn_like(latents)
                    noisy_latents = self.sampler.add_noise(
                        latents=latents, noise=sampler_noise, timesteps=timesteps
                    )

                    # UNet inference
                    noise_pred = self.unet(noisy_latents, context, timesteps)
                    # MSE loss
                    loss = F.mse_loss(noise_pred, sampler_noise)

                # backprop, optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.unet.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.optimizer.step()

                # loss logging
                if step % self.log_interval == 0:
                    self.losses.append(loss.item())

            # save checkpoints every epoch
            finetune_state = {
                name: param.detach().to(device=torch.device("cpu"))
                for name, param in self.unet.named_parameters()
                if param.requires_grad
            }
            checkpoint_path = self.run_dir / "checkpoints" / f"checkpoint-{epoch}.pt"
            torch.save(finetune_state, checkpoint_path)

        # save losses after training
        losses_path = self.run_dir / "losses.json"
        payload = {"log_interval": self.log_interval, "losses": self.losses}
        with open(losses_path, "w") as f:
            json.dump(payload, f)
