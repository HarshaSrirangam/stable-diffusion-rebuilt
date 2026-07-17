"""
Downloads the following (skips existing):
    1. StableDiffusion 1.5 checkpoint -> data/weights
    2. Persian miniature painting raw images from Wikimedia commons
    (entire tree) -> data/persian/raw/

Usage:
    uv run python scripts/download_data.py                 # all
    uv run python scripts/download_data.py --no-ckpt       # no SD1.5 checkpoint
    uv run python scripts/download_data.py --no-persian    # no Persian mini paintings
"""

import io
import time
from pathlib import Path
from collections import deque
import argparse
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm


API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "sdrebuilt-persian/1.0 (https://github.com/HarshaSrirangam/stable-diffusion-rebuilt)"}

def api_get(params, retries=8):
    # wikimedia rate-limits and sometimes returns html instead of json
    # retry with exponential backoff until parseable json is returned
    for attempt in range(retries):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        try:
            return r.json()
        except ValueError:
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"api_get failed after {retries} retries: {params}")

def category_members(category, cmtype):
    # list every member of a category (files or subcategoriess), paginating with continue-tokens
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
        time.sleep(0.3)
    return members

def collect_files(root, max_depth=5):
    # BFS the whole category tree
    # skip (dont crash on) a category that stays throttled through all retries
    titles = []
    seen = set()
    queue = deque([(root, 0)])
    while queue:
        category, depth = queue.popleft()
        if category in seen:
            continue
        seen.add(category)
        try:
            titles += category_members(category, cmtype="file")
            if depth < max_depth:
                for sub in category_members(category, cmtype="subcat"):
                    queue.append((sub, depth + 1))
        except RuntimeError as e:
            print("  skip category", category, e)
            time.sleep(5)
    return list(dict.fromkeys(titles))

def image_url(title, width=768):
    # ask the api for a downscaled thumbnail url
    info = api_get({
        "action": "query", "titles": title,
        "prop": "imageinfo", "iiprop": "url", "iiurlwidth": width,
        "format": "json",
    })
    ii = next(iter(info["query"]["pages"].values()))["imageinfo"][0]
    return ii.get("thumburl", ii["url"])

def download_bytes(url, retries=5):
    # retry with backoff
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            return r.content
        time.sleep(2 ** attempt)
    return None


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

def download_persian():
    raw_dir = Path("data/persian/raw")
    if raw_dir.exists() and any(raw_dir.glob("*.jpg")):
        print(f"Skipping persian")
        return
    print(f"Downloading persian")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # walk the entire "Persian miniatures" category tree
    print("listing persian miniature images from Wikimedia Commons")
    titles = collect_files("Category:Persian miniatures")
    print(f"pool size after dedupe: {len(titles)}")

    # download every image that passes the 256px filter
    print(f"Downloading {len(titles)} images")
    idx = 0
    for title in tqdm(titles, desc="downloading"):
        try:
            if not title.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            raw = download_bytes(image_url(title))
            if raw is None:
                print("  skip", title, "bad response")
                continue
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if min(img.size) < 256:
                continue
            img.save(raw_dir / f"{idx:04d}.jpg")
            idx += 1
        except Exception as e:
            print("  skip", title, e)
        time.sleep(0.1)
    print(f"Saved {idx} images to {raw_dir}")

def main(ckpt=True, persian=True):
    if ckpt:
        download_checkpoint()
    if persian:
        download_persian()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persian", action=argparse.BooleanOptionalAction, default=True)
    args=parser.parse_args()
    main(
        ckpt=args.ckpt,
        persian=args.persian
    )