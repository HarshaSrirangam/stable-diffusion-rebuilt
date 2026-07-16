"""
Downloads the following (skips existing):
    1. StableDiffusion 1.5 checkpoint
    2. persian miniature image dataset + BLIP captions

Usage:
    uv run scripts/download_data.py
"""

import io
import json
import time
from pathlib import Path
from collections import deque
import argparse

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from tqdm import tqdm


API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "sdrebuilt-persian/1.0 (https://github.com/HarshaSrirangam/stable-diffusion-rebuilt)"}

def api_get(params, retries=5):
    # wikimedia rate-limits + sometimes returns html instead of json.
    # retry with exponential backoff until we get parseable json
    for attempt in range(retries):
        r = requests.get(API, params=params, headers=HEADERS)
        try:
            return r.json()
        except ValueError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"api_get failed after {retries} retries: {params}")

def category_members(category, cmtype):
    # list every member of a category (files or subcats), paginating through continue-tokens
    members = []
    cont = {}
    while True:
        r = api_get({
            "action": "query", "list": "categorymembers",
            "cmtitle": category, "cmtype": cmtype,
            "cmlimit": "500", "format": "json", **cont,
        })
        members += [m["title"] for m in r["query"]["categorymembers"]]
        cont = r.get("continue")
        if not cont:
            break
        time.sleep(0.1)
    return members

def collect_files(root, n_images, max_depth=5):
    # BFS over the category tree: grab files at each level, descend into subcats
    titles = []
    seen = set()
    queue = deque([(root, 0)])
    while queue and len(titles) < n_images:
        category, depth = queue.popleft()
        if category in seen:
            continue
        seen.add(category)

        titles += category_members(category, cmtype="file")
        if depth < max_depth:
            for sub in category_members(category, cmtype="subcat"):
                queue.append((sub, depth + 1))
    return titles[:n_images]

def image_url(title):
    # resolve a File: title to its actual download url
    info = api_get({
        "action": "query", "titles": title,
        "prop": "imageinfo", "iiprop": "url", "format": "json",
    })
    return next(iter(info["query"]["pages"].values()))["imageinfo"][0]["url"]


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

def download_persian(n_images=1200):
    destination = Path("data/persian")
    if (destination / "metadata.jsonl").exists():
        print(f"Skipping persian")
        return
    print(f"Downloading persian")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(parents=True, exist_ok=True)

    # walk the "Persian miniatures" category tree for image filenames
    print("listing persian miniature images from Wikimedia Commons")
    titles = collect_files("Category:Persian miniatures", n_images)

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
        for title in tqdm(titles, desc="downloading + captioning"):
            try:
                url = image_url(title)

                raw = requests.get(url, headers=HEADERS).content
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

def main(ckpt=True, persian=True, persian_count=1200):
    if ckpt:
        download_checkpoint()
    if persian:
        download_persian(n_images=persian_count)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persian", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persian-count", type=int, default=1200)
    args=parser.parse_args()
    main(
        ckpt=args.ckpt,
        persian=args.persian,
        persian_count=args.persian_count
    )