"""
Evaluates a single LoRA run. Generates images on a fixed set of eval prompts and
computes validation metrics. Outputs written to runs/<run>/eval/.

Outputs:
    grid.png        LoRA generations for each eval prompt (laid out in a grid).
    prompts.txt     Eval prompts.
    loss_curve.png  Training loss over batches (from losses.json).
    metrics.json    The metrics below.

Metrics:
    val_loss_ratio          Noise-prediction MSE on the eval dataset, lora vs base model
                            (ratio = LoRA / base). Below 1 means the LoRA adapter
                            predicts noise added to persian images better than the base
                            model does.
    clip_prompt_adherence   Average cosine similarity between CLIP embeddings of each
                            generated image (on eval prompts) and their prompts.
    clip_style_adherence    Average cosine similarity between CLIP embeddings of each
                            generated image and the mean embedding of the eval images.

Note:
    CLIP prompt adherence and style adherence will likely be systematically
    worse and systematically better, respectively, with LoRA compared to the
    base model. These validation metrics are included for the purpose of
    comparing different LoRA runs, not necessarily for comparing LoRA to the
    base model. They can still be used to check that a run hasn't veered off,
    though inspecting grid.png would be easier.

Usage:
    uv run python scripts/evaluate_lora.py --run <run_name>
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import math
from pathlib import Path

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from transformers.utils import logging as hf_logging

from sdrebuilt.convert_weights import load_all
from sdrebuilt.dataset import precompute
from sdrebuilt.inference import InferencePipeline
from sdrebuilt.lora.utils import disable_lora, enable_lora, inject_lora
from sdrebuilt.model.autoencoder import Autoencoder
from sdrebuilt.model.clip import CLIP
from sdrebuilt.model.unet import UNet
from sdrebuilt.samplers.ddim import DDIM
from sdrebuilt.samplers.ddpm import DDPM

datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()
hf_logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCH_SIZE = 4
PROMPTS = [
    "a man on a horse",
    "two people in a garden",
    "a king on a throne",
    "a battle between warriors",
    "a woman playing a musical instrument",
    "a hunter shooting a bow",
    "people feasting in a palace",
    "a man reading a book under a tree",
    "a caravan of camels",
    "two lovers meeting at night",
]


def log(msg: str) -> None:
    print(f"\n>>> {msg}")


def val_loss(
    unet: UNet,
    loader: DataLoader,
    sampler: DDPM | DDIM,
    device: torch.device,
    seed: int,
) -> float:
    # seed so lora and base passes have same random timesteps
    torch.manual_seed(seed)
    unet.to(device).eval()
    running_loss = 0
    running_samples = 0.0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for latents, context in loader:
            latents, context = latents.to(device), context.to(device)
            b = latents.shape[0]
            # sample random timesteps
            timesteps = torch.randint(low=0, high=sampler.n_step_train, size=(b,)).to(
                device=device, dtype=torch.long
            )
            # sample and add noise to latents
            sampler_noise = torch.randn_like(latents)
            noisy_latents = sampler.add_noise(
                latents=latents, noise=sampler_noise, timesteps=timesteps
            )
            # UNet inference
            noise_pred = unet(noisy_latents, context, timesteps)
            # MSE loss
            loss = F.mse_loss(noise_pred, sampler_noise, reduction="mean")
            running_loss += loss.item() * b  # total batch loss
            running_samples += b
    return running_loss / running_samples


def main(run_dir: Path):
    # load run training config
    lora_cfg_path = run_dir / "training_config.yaml"
    with open(lora_cfg_path, "r") as f:
        lora_cfg = yaml.safe_load(f)

    # create output dir
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        raise FileExistsError(f"{run_dir}/eval/ already exists")
    eval_dir.mkdir()

    # 1) loss curve (loss_curve.png)
    loss_path = run_dir / "losses.json"
    loss_curve_path = eval_dir / "loss_curve.png"
    with open(loss_path, "r") as f:
        loss = json.load(f)
    log_interval = loss["log_interval"]
    losses = loss["losses"]
    x = [i * log_interval for i in range(len(losses))]
    plt.plot(x, losses)
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) frozen copy of prompts (prompts.txt)
    prompts_path = eval_dir / "prompts.txt"
    with open(prompts_path, "w") as f:
        for p in PROMPTS:
            f.write(p + "\n")

    # 3) lora inference on eval prompts (grid.png)
    torch.manual_seed(SEED)
    grid_path = eval_dir / "grid.png"
    # load model
    log("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = Autoencoder().eval()
    clip = CLIP().eval()
    unet = UNet().eval()
    load_all(ROOT / lora_cfg["pretrained_path"], vae=vae, clip=clip, unet=unet)
    # inject LoRA layers
    log("Loading LoRA adapter")
    inject_lora(
        model=unet,
        target_names=lora_cfg["targets"]["layers"],
        r=lora_cfg["r"],
        alpha=lora_cfg["alpha"],
    )
    # load last LoRA checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpts = ckpt_dir.glob("checkpoint-*.pt")
    ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("-")[1]))
    lora_state = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(lora_state, strict=False)
    # sampler
    sampler = DDPM()
    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # generate eval images -> grid
    log("Generating eval images")
    inference_pipeline = InferencePipeline(
        vae=vae,
        clip=clip,
        unet=unet,
        sampler=sampler,
        tokenizer=tokenizer,
        device=device,
        idle_device="cpu",
    )
    grid_imgs = []  # np arrays
    for p in PROMPTS:
        grid_imgs.append(inference_pipeline.txt_2_img(prompt=p, seed=SEED))
    n = len(grid_imgs)
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax, img, p in zip(axes[:n], grid_imgs, PROMPTS, strict=True):
        ax.imshow(img)
        ax.set_title(p)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4) eval metrics
    # i) CLIP prompt adherence
    log("Computing CLIP prompt adherence")
    clip_model = (
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    )
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        inp = clip_proc(
            images=grid_imgs, text=PROMPTS, return_tensors="pt", padding=True
        ).to(device)
        out = clip_model(**inp)
        img_emb = out.image_embeds  # (n_prompts, d_embed)
        txt_emb = out.text_embeds  # (n_prompts, d_embed)
        # cosine similarity
        dot_prods = (img_emb * txt_emb).sum(dim=1)  # (n_prompts,)
        img_mag = img_emb.norm(dim=-1, keepdim=False)  # (n_prompts,)
        txt_mag = txt_emb.norm(dim=-1, keepdim=False)  # (n_prompts,)
        cosines = dot_prods / (img_mag * txt_mag)  # (n_prompts,)
        prompt_adherence = cosines.mean().item()

    # ii) CLIP style adherence
    log("Computing CLIP style adherence")
    eval_imgs_path = ROOT / "data/persian/eval/images"
    paths = sorted((eval_imgs_path).glob("*.jpg"))  # all ~150 image paths
    eval_imgs = [Image.open(p).convert("RGB") for p in paths]
    with torch.no_grad():
        inp = clip_proc(
            images=eval_imgs,
            return_tensors="pt",
        ).to(device)
        eval_embs = clip_model.get_image_features(**inp)  # (len(eval_imgs), d_embed)
        eval_emb = (eval_embs / eval_embs.norm(dim=-1, keepdim=True)).mean(
            0
        )  # (d_embed,)
        # cosine similarity
        dot_prods = (img_emb * eval_emb.unsqueeze(0)).sum(dim=1)  # (n_prompts,)
        eval_mag = eval_emb.norm(dim=-1, keepdim=True)  # (1,)
        cosines = dot_prods / (eval_mag * img_mag)  # (n_prompts,)
        style_adherence = cosines.mean().item()

    # iii) eval dataset noise pred
    log("Running UNet on eval dataset")
    cache_path = ROOT / "data" / "cache" / f"{lora_cfg['dataset']}_eval.pt"
    if not cache_path.exists():
        log("Precomputing image/caption embeddings (no cache found)")
        precompute(
            pretrained_path=ROOT / lora_cfg["pretrained_path"],
            dataset=lora_cfg["dataset"],
            split="eval",
            batch_size=lora_cfg["batch_size"],
            device=lora_cfg["device"],
            cache_path=cache_path,
        )
    data = torch.load(cache_path)
    dataset = TensorDataset(data["latents"], data["context"])
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # inference
    enable_lora(unet)
    lora_loss = val_loss(
        unet=unet, loader=eval_loader, sampler=sampler, device=device, seed=SEED
    )
    disable_lora(unet)
    base_loss = val_loss(
        unet=unet, loader=eval_loader, sampler=sampler, device=device, seed=SEED
    )
    ratio = lora_loss / base_loss
    # write eval results to metrics.json
    metrics = {
        "val_loss_base": base_loss,
        "val_loss_lora": lora_loss,
        "val_loss_ratio": ratio,
        "clip_prompt_adherence": prompt_adherence,
        "clip_style_adherence": style_adherence,
    }
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Done. Eval outputs written to {eval_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()
    main(run_dir=ROOT / "runs" / args.run)
