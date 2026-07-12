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
from torch.utils.data import TensorDataset, DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm

from sdrebuilt.convert_weights import load_vae, load_clip, load_unet
from sdrebuilt.dataset import ImageCaptionDataset
from sdrebuilt.lora.utils import inject_lora
from sdrebuilt.model.autoencoder import Autoencoder
from sdrebuilt.model.clip import CLIP
from sdrebuilt.model.unet import UNet
from sdrebuilt.samplers.ddpm import DDPM
from sdrebuilt.trainer import Trainer

@torch.no_grad()
def precompute(config, cache_path: Path) -> None:
    """
    Precomputes and saves images->latents and captions->clip embeddings to disk. 
    """
    # build frozen encoders
    device = config["device"]
    pretrained_path = config["pretrained_path"]
    vae = Autoencoder().eval().requires_grad_(False)
    load_vae(
        path=pretrained_path,
        vae=vae
    )
    vae.to(device)
    clip = CLIP().eval().requires_grad_(False)
    load_clip(
        path=pretrained_path,
        clip=clip
    )
    clip.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # build raw dataset from config
    dataset = ImageCaptionDataset(config["dataset"], image_size=512)
    loader = DataLoader(dataset, batch_size=config["batch_size"])

    # encode data
    latents_list, context_list = [], []
    pbar = tqdm(loader, desc="Encoding image/caption batches", colour="blue")
    for batch in pbar:
        images = batch["image"].to(device)
        captions = batch["caption"]
        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device=device, dtype=torch.long)
        encoder_noise = torch.randn((images.size(0), 4, 64, 64), device=device)
        latents_list.append(vae.encode(images, encoder_noise).cpu())
        context_list.append(clip(tokens).cpu())
    vae.to("cpu")
    clip.to("cpu")
    torch.cuda.empty_cache()

    # save embeddings to disk
    latents = torch.cat(latents_list)  # (B, 4, 64, 64)
    context = torch.cat(context_list)  # (B, 77, 768)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "context": context}, cache_path)

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

    # load UNet and inject LoRA layers
    print("Loading UNet weights...")
    device = torch.device(config["device"])  # cuda
    unet = UNet().eval().requires_grad_(False)
    load_unet(
        path=config["pretrained_path"],
        unet=unet,
    )
    unet.to(device)
    print("Injecting LoRA layers...")
    unet.requires_grad_(False)
    inject_lora(
        model=unet,
        target_names=config["targets"]["layers"],
        r=config["r"],
        alpha=config["alpha"],
    )
    # build dataset and dataloader
    print("Preparing dataset...")
    cache_path = ROOT / "data" / "cache" / f"{config['dataset']['name']}.pt"
    if not cache_path.exists():
        precompute(config=config, cache_path=cache_path)
    data = torch.load(cache_path)
    dataset = TensorDataset(data["latents"], data["context"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

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
        unet=unet,
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
