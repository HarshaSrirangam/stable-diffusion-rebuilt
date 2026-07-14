"""
Generate images (base model or with LoRA adapters)

Reads configs/inference.yaml, builds the inference pipeline (optionally injecting
lora adapters from a specific run), and runs txt2img or img2img based on whether
an input image path is provided in the config. Writes output.png (and input.png for
img2img) plus a config snapshot to runs/<run>/generated_images/<name>/.

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
    ROOT = Path(__file__).resolve().parents[1]
    config_path = ROOT / "configs" / "inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # create output directory
    run_name = config["run"]
    lora_enabled = run_name is not None
    if lora_enabled:
        run_dir = ROOT / "runs" / run_name
    else:
        run_dir = ROOT / "base_model"
    (run_dir / "generated_images").mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "generated_images" / config["name"]
    if output_dir.exists():
        raise FileExistsError(f"output '{config['name']}' already exists")
    output_dir.mkdir(parents=True)
    # freeze config
    shutil.copy(config_path, output_dir / "inference_config.yaml")

    # seed
    torch.manual_seed(config["seed"])

    # load models (keep on cpu for now)
    log("Loading model")
    device = config["device"]
    vae = Autoencoder().eval()
    clip = CLIP().eval()
    unet = UNet().eval()
    load_all(config["pretrained_path"], vae=vae, clip=clip, unet=unet)

    # retrieve lora config and inject layers (if lora enabled)
    if lora_enabled:
        log("Loading LoRA adapter")
        lora_cfg_path = run_dir / "training_config.yaml"
        with open(lora_cfg_path, "r") as f:
            lora_config = yaml.safe_load(f)
        inject_lora(
            model=unet,
            target_names=lora_config["targets"]["layers"],
            r=lora_config["r"],
            alpha=lora_config["alpha"],
        )
        # load checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if config["checkpoint"] == "last":
            ckpts = ckpt_dir.glob("checkpoint-*.pt")
            lora_ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("-")[1]))
        else:
            lora_ckpt_path = ckpt_dir / f"checkpoint-{config['checkpoint']}.pt"
        lora_state = torch.load(lora_ckpt_path, map_location="cpu")
        unet.load_state_dict(lora_state, strict=False)

    # sampler
    SAMPLERS = {"ddpm": DDPM, "ddim": DDIM}
    sampler = SAMPLERS[config["sampler"]["name"]](
        n_step_inf=config["sampler"]["n_step_inf"]
    )

    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # InferencePipeline
    log("Generating")
    inference_pipeline = InferencePipeline(
        vae=vae,
        clip=clip,
        unet=unet,
        sampler=sampler,
        tokenizer=tokenizer,
        device=device,
        idle_device="cpu",
    )
    if config["input_image_path"] is None:  # txt2img
        generated_image = inference_pipeline.txt_2_img(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            seed=config["seed"],
        )
    else:  # img2img
        input_image = Image.open(ROOT / config["input_image_path"])
        input_image = input_image.convert("RGB").resize((512, 512))
        input_image_arr = np.array(input_image, dtype=np.uint8)

        # turn into raise errors
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

    # save outputs
    generated_image.save(output_dir / "output.png")
    if config["input_image_path"] is not None:
        input_image.save(output_dir / "input.png")


if __name__ == "__main__":
    main()
