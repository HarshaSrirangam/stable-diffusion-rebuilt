"""
Generate images with base model or with LoRA adapters. Reads configs/inference.yaml
and runs txt2img or img2img based on whether an input image path is provided
in the config. Writes output.png (and input.png for img2img) plus a snapshot
of the inference config to the resolved output directory.

Usage:
    uv run python scripts/generate.py
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import shutil
from pathlib import Path

import huggingface_hub
import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file, safe_open
from transformers import CLIPTokenizer
from transformers.utils import logging as hf_logging

from sdrebuilt.convert_weights import load_all
from sdrebuilt.inference import InferencePipeline
from sdrebuilt.lora.utils import inject_lora
from sdrebuilt.model.autoencoder import Autoencoder
from sdrebuilt.model.clip import CLIP
from sdrebuilt.model.unet import UNet
from sdrebuilt.samplers.ddim import DDIM
from sdrebuilt.samplers.ddpm import DDPM

hf_logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"\n>>> {msg}")


def main():
    # load inference config
    config_path = ROOT / "configs" / "inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # create output directory
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    used = [int(d.name[3:]) for d in outputs.glob("out*") if d.name[3:].isdigit()]
    out_dir = outputs / f"out{max(used, default=0) + 1}"
    out_dir.mkdir()
    shutil.copy(config_path, out_dir / "inference_config.yaml")

    # seed
    torch.manual_seed(config["seed"])

    # load models (keep on cpu for now)
    log("Loading base model")
    device = config["device"]
    vae = Autoencoder().eval()
    clip = CLIP().eval()
    unet = UNet().eval()
    load_all(ROOT / config["pretrained_path"], vae=vae, clip=clip, unet=unet)

    # retrieve lora config and inject layers (if lora enabled)
    if config["lora_path"]:
        # load persian lora weights and metadata
        log("Loading LoRA adapter")
        lora_path = ROOT / config["lora_path"]
        lora_state = load_file(lora_path)
        with safe_open(lora_path, framework="pt") as f:
            meta = f.metadata()
        r, alpha = int(meta["r"]), int(meta["alpha"])
        targets = meta["targets"].split(",")
        # inject layers
        inject_lora(
            model=unet,
            target_names=targets,
            r=r,
            alpha=alpha,
        )
        unet.load_state_dict(lora_state, strict=False)

    # sampler
    SAMPLERS = {"ddpm": DDPM, "ddim": DDIM}
    sampler = SAMPLERS[config["sampler"]](n_step_inf=config["n_steps"])

    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # InferencePipeline
    log("Generating image")
    inference_pipeline = InferencePipeline(
        vae=vae,
        clip=clip,
        unet=unet,
        sampler=sampler,
        tokenizer=tokenizer,
        device=device,
        idle_device="cpu",
    )
    if config["input_image"] is None:  # txt2img
        generated_image = inference_pipeline.txt_2_img(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            seed=config["seed"],
        )
    else:  # img2img
        input_image = Image.open(ROOT / config["input_image"])
        input_image = input_image.convert("RGB").resize((512, 512))
        input_image_arr = np.array(input_image, dtype=np.uint8)

        if input_image_arr.shape != (512, 512, 3):
            raise ValueError("must be (512, 512, 3)")
        if input_image_arr.dtype != np.uint8:
            raise ValueError("must be uint8")
        if input_image_arr.min() < 0 or input_image_arr.max() > 255:
            raise ValueError("must be in [0, 255]")

        generated_image = inference_pipeline.img_2_img(
            input_image=input_image_arr,
            strength=config["strength"],
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            seed=config["seed"],
        )
    log("Saving output")
    generated_image = Image.fromarray(generated_image)

    # save outputs to out dir
    generated_image.save(out_dir / "output.png")
    if config["input_image"] is not None:
        input_image.save(out_dir / "input.png")
    log(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
