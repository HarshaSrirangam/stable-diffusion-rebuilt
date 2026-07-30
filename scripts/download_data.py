"""Download SD1.5 checkpoint and persian miniature lora from HF into weights/."""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "weights"

def main(base: bool, lora: bool):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if base:
        print("Downloading SD1.5 checkpoint")
        hf_hub_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                        "v1-5-pruned-emaonly.safetensors", local_dir=WEIGHTS)
    if lora:
        print("Downloading persian miniature lora weights")
        hf_hub_download("HarshaSrirangam/persian-miniature-lora",
                        "persian_lora.safetensors", local_dir=WEIGHTS)
    print("Done. Saved to data/weights/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(
        base=args.base, lora=args.lora
    )
