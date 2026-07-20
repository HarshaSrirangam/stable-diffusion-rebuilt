"""
Train a LoRA adapter.

Reads configs/lora.yaml, freezes base model, injects LoRA layers into
the UNet (target layers specified in config), and trains on the config dataset.
Writes checkpoints/, losses.json, and training_config.yaml to runs/<run_name>.

Usage:
    uv run python scripts/train_lora.py
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import shutil
from pathlib import Path

import datasets
import huggingface_hub
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPTokenizer
from transformers.utils import logging as hf_logging

from sdrebuilt.convert_weights import load_clip, load_unet, load_vae
from sdrebuilt.dataset import precompute
from sdrebuilt.lora.utils import inject_lora
from sdrebuilt.model.autoencoder import Autoencoder
from sdrebuilt.model.clip import CLIP
from sdrebuilt.model.unet import UNet
from sdrebuilt.samplers.ddpm import DDPM
from sdrebuilt.trainer import Trainer

hf_logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"\n>>> {msg}")


def main():
    # load lora config
    config_path = ROOT / "configs" / "lora.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # create run directory
    run_name = (
        f"{config['dataset']}_r{config['r']}"
        f"_{config['targets']['desc']}_{config['name']}"
    )
    run_dir = ROOT / "runs" / run_name
    frozen_cfg = run_dir / "training_config.yaml"
    if run_dir.exists():
        existing_frozen_cfg = yaml.safe_load(open(frozen_cfg))
        if config == existing_frozen_cfg:
            print(
                f">>> run '{run_name}' already trained with identical config. Skipping."
            )
            return
        raise FileExistsError(
            f"run '{run_dir.name}' already exists with a DIFFERENT config. "
            f"Modify 'name' field in lora.yaml or delete folder to retrain."
        )
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    # freeze config
    shutil.copy(config_path, frozen_cfg)

    # seed
    torch.manual_seed(config["seed"])

    # load UNet and inject LoRA layers
    log("Loading UNet and injecting LoRA layers")
    device = torch.device(config["device"])  # cuda
    unet = UNet().eval().requires_grad_(False)
    load_unet(
        path=ROOT / config["pretrained_path"],
        unet=unet,
    )
    unet.to(device)
    unet.requires_grad_(False)
    inject_lora(
        model=unet,
        target_names=config["targets"]["layers"],
        r=config["r"],
        alpha=config["alpha"],
    )
    # build dataset and dataloader
    log("Preparing dataset")
    cache_path = ROOT / "data" / "cache" / f"{config['dataset']}_train.pt"
    if not cache_path.exists():
        log("Precomputing image/caption embeddings (no cache found)")
        precompute(
            pretrained_path=ROOT / config["pretrained_path"],
            dataset=config["dataset"],
            split="train",
            batch_size=config["batch_size"],
            device=config["device"],
            cache_path=cache_path
        )
    data = torch.load(cache_path)
    dataset = TensorDataset(data["latents"], data["context"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # optimizer
    optimizer = optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=config["lr"]
    )

    # noise sampler
    sampler = DDPM()  # 1000 train steps by default

    # Trainer (training loop)
    trainer = Trainer(
        unet=unet,
        dataloader=train_loader,
        optimizer=optimizer,
        sampler=sampler,
        device=device,
        n_epochs=config["n_epochs"],
        log_interval=config["log_interval"],
        run_dir=run_dir,
    )
    log("Training")
    trainer.train()


if __name__ == "__main__":
    main()
    log("Done")