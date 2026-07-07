"""
Train LoRA adapters.

Reads configs/lora.yaml, freezes base model, injects LoRA layers into
the UNet (as specified by config), and trains on the configured dataset.
Writes checkpoints/, losses.json, and config.yaml to runs/<run_name>.

Usage:
    uv run python scripts/train_lora.py
"""

import shutil
from pathlib import Path

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from sdrebuilt.convert_weights import load_all
from sdrebuilt.dataset import ImageCaptionDataset
from sdrebuilt.lora.utils import inject_lora
from sdrebuilt.model.autoencoder import Autoencoder
from sdrebuilt.model.clip import CLIP
from sdrebuilt.model.unet import UNet
from sdrebuilt.samplers.ddpm import DDPM
from sdrebuilt.trainer import Trainer


def main():
    # load lora config
    ROOT = Path(__file__).resolve().parents[1]
    config_path = ROOT / "configs" / "lora.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # create run directory
    run_name = f"{config['dataset']['name']}_r{config['r']}_{config['targets']['desc']}_{config['name']}"
    run_dir = ROOT / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"run '{run_dir.name}' already exists")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    # freeze config
    shutil.copy(config_path, run_dir / "training_config.yaml")

    # seed
    torch.manual_seed(config["seed"])

    # load sd models
    print("Loading pretrained weights...")
    device = torch.device(config["device"])  # cuda
    vae = Autoencoder().to(device=device)
    clip = CLIP().to(device=device)
    unet = UNet().to(device=device)
    load_all(config["pretrained_path"], vae=vae, clip=clip, unet=unet)

    # inject lora layers
    print("Injecting LoRA layers...")
    vae.requires_grad_(False)
    clip.requires_grad_(False)
    unet.requires_grad_(False)
    inject_lora(
        model=unet,
        target_names=config["targets"]["layers"],
        r=config["r"],
        alpha=config["alpha"],
    )

    # build dataset and dataloader
    print("Preparing dataset...")
    train_dataset = ImageCaptionDataset(source=config["dataset"], image_size=512)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # optimizer
    optimizer = optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=config["lr"]
    )

    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # noise sampler
    sampler = DDPM()  # 1000 train steps by default

    # Trainer (training loop)
    trainer = Trainer(
        vae=vae,
        clip=clip,
        unet=unet,
        tokenizer=tokenizer,
        dataloader=train_loader,
        optimizer=optimizer,
        sampler=sampler,
        device=device,
        n_epochs=config["n_epochs"],
        log_interval=config["log_interval"],
        run_dir=run_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
    print("Training finishes")
