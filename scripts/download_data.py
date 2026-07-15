"""
Downloads the following (skips existing):
    1. StableDiffusion 1.5 chekcpoint
    2. ex-libris image dataset + BLIP captions

Usage:
    uv run scripts/download_data.py
"""

import io
import json
import time
from pathlib import Path
import argparse

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def download_checkpoint():
    destination = Path("data/weights")
    filename = "v1-5-pruned-emaonly.safetensors"
    target = destination / filename
    if target.exists():
        print(f"Skipping SD1.5 checkpoint")
        return
    print(f"Downloading checkpoint")
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    destination.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(destination)
    )

def download_exlibris(n_images=1200):
    destination = Path("data/exlibris")
    if (destination / "metadata.jsonl").exists():
        print(f"Skipping exlibris")
        return
    print(f"Downloading exlibris")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(parents=True, exist_ok=True)

    # list image filenames in the "Bookplates" category
    commons_api = "https://commons.wikimedia.org/w/api.php"
    print("listing ex-libris images from Wikimedia Commons")
    titles = []
    cont = {}
    while len(titles) < n_images:
        r = requests.get(commons_api, params={
            "action": "query", "list": "categorymembers",
            "cmtitle": "Category:Bookplates", "cmtype": "file",
            "cmlimit": "500", "format": "json", **cont,
        }).json()
        titles += [m["title"] for m in r["query"]["categorymembers"]]
        cont = r.get("continue")
        if not cont:
            break
    titles = titles[:n_images]

    # load BLIP
    print("loading BLIP")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    # download each image, generate caption with BLIP, save image and metadata
    print(f"Downloading + captioning up to {len(titles)} images")
    idx = 0
    with open(destination / "metadata.jsonl", "w") as meta:
        for title in titles:
            try:
                info = requests.get(commons_api, params={
                    "action": "query", "titles": title,
                    "prop": "imageinfo", "iiprop": "url", "format": "json",
                }).json()
                url = next(iter(info["query"]["pages"].values()))["imageinfo"][0]["url"]

                raw = requests.get(url, headers={"User-Agent": "exlibris-prep"}).content
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if min(img.size) < 256:
                    continue

                inp = processor(img, return_tensors="pt").to(device)
                cap = processor.decode(blip.generate(**inp, max_new_tokens=30)[0], skip_special_tokens=True)

                fname = f"images/{idx:04d}.jpg"
                img.save(destination / fname)
                meta.write(json.dumps({"file_name": fname, "text": cap}) + "\n")
                idx += 1
            except Exception as e:
                print("  skip", title, e)
            time.sleep(0.1)
    print(f"Saved {idx} image/caption pairs to {destination}")

def main(ckpt=True, exlibris=True, exlibris_count=1200):
    if ckpt:
        download_checkpoint()
    if exlibris:
        download_exlibris(n_images=exlibris_count)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exlibris", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exlibris-count", type=int, default=1200)
    args=parser.parse_args()
    main(
        ckpt=args.ckpt,
        exlibris=args.exlibris,
        exlibris_count=args.exlibris_count
    )