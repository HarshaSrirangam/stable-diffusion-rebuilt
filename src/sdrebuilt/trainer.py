import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm
from pathlib import Path
import json

from .samplers.ddpm import DDPM
from .model.autoencoder import Autoencoder
from .model.clip import CLIP
from .model.unet import UNet


class Trainer:
    """
    Trainer class.

    Args:
        vae: frozen autoencoder
        clip: frozen clip text embedder
        unet: LoRA-injected UNet, all but LoRA layers frozen
        tokenizer: CLIP tokenizer
        dataloader: DataLoader of batched image-caption pairs
        optimizer: optim.Optimizer,
        scheduler: DDPM sampler
        device: training device
    """
    def __init__(
        self,
        vae: Autoencoder,
        clip: CLIP,
        unet: UNet,
        tokenizer: CLIPTokenizer,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: DDPM,
        device: torch.device,
        n_epochs: int,
        log_interval: int,
        run_dir: Path
    ):
        self.vae = vae
        self.clip = clip
        self.unet = unet
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.run_dir = run_dir

        self.losses = []

    def train(self):
        self.vae.eval()
        self.clip.eval()
        self.unet.train()
        for epoch in range(self.n_epochs):
            pbar = tqdm(self.dataloader, desc=f"epoch {epoch + 1}/{self.n_epochs}", colour="blue")
            for step, batch in enumerate(pbar):
                images = batch["image"].to(device=self.device) # (B, 3, 512, 512) tensor
                captions = batch["caption"] # list of B num of strings
                b = images.shape[0]

                # encode images
                encoder_noise = torch.randn((b, 4, 64, 64), device=self.device)
                with torch.no_grad():
                    latents = self.vae.encode(images, encoder_noise)

                # tokenize captions
                tokens = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].to(device=self.device, dtype=torch.long)

                # encode caption tokens
                with torch.no_grad():
                    captions_emb = self.clip(tokens)

                # sample random timesteps in [0, 1000]
                timesteps = torch.randint(
                    low=0,
                    high=self.scheduler.n_step_train,
                    size=(b,)
                ).to(device=self.device, dtype=torch.long)

                # sample and add noise to latents
                scheduler_noise = torch.randn_like(latents)
                latents = self.scheduler.add_noise(
                    latents=latents,
                    noise=scheduler_noise,
                    timesteps=timesteps
                )

                # predict noise
                noise_pred = self.unet(latents, captions_emb, timesteps)

                # MSE loss
                loss = F.mse_loss(noise_pred, scheduler_noise)

                # backprop, optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss logging
                if step % self.log_interval == 0:
                    self.losses.append(loss.item())

            # save LoRA checkpoints every epoch
            lora_state = {
                name: param.detach().to(device=torch.device("cpu"))
                for name, param in self.unet.named_parameters() if param.requires_grad
            }
            checkpoint_path = self.run_dir / "checkpoints" / f"checkpoint-{epoch}.pt"
            torch.save(lora_state, checkpoint_path)
        
        # save losses after training
        losses_path = self.run_dir / "losses.json"
        payload = {
            "log_interval": self.log_interval,
            "losses": self.losses
        }
        with open(losses_path, "w") as f:
            json.dump(payload, f)
