import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPTokenizer

from .model.autoencoder import Autoencoder
from .model.clip import CLIP
from .model.unet import UNet
from .samplers.ddpm import DDPM


class Trainer:
    """
    Finetune method-agnostic trainer class. Supports UNet finetuning only.

    Args:
        vae: frozen autoencoder
        clip: frozen clip text embedder
        unet: UNet with only intended trainable params active
        tokenizer: CLIP tokenizer
        dataloader: DataLoader of batched image-caption pairs
        optimizer: Adam or AdamW
        sampler: DDPM sampler
        device: training device
        n_epochs: number of training epochs
        log_interval: number of batches between logging loss
        run_dir: runs/<current_run>
    """

    def __init__(
        self,
        vae: Autoencoder,
        clip: CLIP,
        unet: UNet,
        tokenizer: CLIPTokenizer,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        sampler: DDPM,
        device: torch.device,
        n_epochs: int,
        log_interval: int,
        run_dir: Path,
    ):
        self.vae = vae
        self.clip = clip
        self.unet = unet
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.sampler = sampler
        self.device = device
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.run_dir = run_dir

        self.losses = []

    @torch.no_grad()
    def _precompute(self):
        """
        Precomputes and caches images->latents and captions->clip embeddings. Offloads
        vae and clip to cpu. Returns cached dataloader.
        """
        self.vae.eval()
        self.clip.eval()
        latents_list, context_list = [], []
        for batch in self.dataloader:
            images = batch["image"].to(device=self.device)
            captions = batch["caption"]
            b = images.shape[0]
            tokens = self.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(device=self.device, dtype=torch.long)

            # encode images and captions
            encoder_noise = torch.randn((b, 4, 64, 64), device=self.device)
            latents_list.append(self.vae.encode(images, encoder_noise).cpu())
            context_list.append(self.clip(tokens).cpu())
        self.vae.to("cpu")
        self.clip.to("cpu")
        torch.cuda.empty_cache()

        latents = torch.cat(latents_list)  # (B, 4, 64, 64)
        context = torch.cat(context_list)  # (B, 77, 768)
        cached_ds = TensorDataset(latents, context)
        return DataLoader(
            cached_ds, batch_size=self.dataloader.batch_size, shuffle=True
        )

    def train(self):
        """
        Run full training loop once.

        Precomputes cached image latents and caption embeddings, then trains
        UNet for n_epochs. Uses bf16 autocast.
        """
        self.unet.train()
        print("Precomputing embeddings...")
        cached_loader = self._precompute()  # create cached dataloader
        print("Training begins")
        for epoch in range(self.n_epochs):
            pbar = tqdm(
                cached_loader, desc=f"epoch {epoch + 1}/{self.n_epochs}", colour="blue"
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
        print("Saving losses...")
        losses_path = self.run_dir / "losses.json"
        payload = {"log_interval": self.log_interval, "losses": self.losses}
        with open(losses_path, "w") as f:
            json.dump(payload, f)
