"""
Generate captions for a folder of images with BLIP and clean them.

Reads every .jpg in data_dir/images, generates a caption for each with BLIP, and
strips the prefix phrases (e.g. "a painting of", "a page of a manuscript with")
to allow LoRA to learn style without trigger words. Run on GPU is preferred. Note
that this script requires data_dir/images to be pre-populated with the outputs
of scrape.py or other images.

Outputs (to data_dir):
    metadata.jsonl          Cleaned BLIP captions.
    metadata_raw.jsonl      Raw BLIP captions.

Usage:
    uv run python scripts/caption.py --data-dir data/persian/<pool>
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import re
from pathlib import Path

import huggingface_hub
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 16  # BLIP batch size
PREFIXES = [
    "a picture taken from a manuscript of",
    "a picture taken from",
    "a page of a manuscript with a painting of",
    "a page of a manuscript with a",
    "a page of a manuscript with",
    "a page of a manuscript",
    "page of a manuscript with a painting of",
    "page of a manuscript with a",
    "page of a manuscript with",
    "page of a manuscript",
    "a page of a book with a painting of",
    "a page of a book with a",
    "a page of a book with",
    "a page of a book",
    "page of a book with a",
    "page of a book",
    "a page of",
    "page of",
    "a book with a picture of",
    "a book with a painting of",
    "a book with a drawing of",
    "a book with a",
    "a book with",
    "a book",
    "a persian manuscript with a painting of",
    "a persian manuscript with a",
    "a persian manuscript with",
    "a persian manuscript",
    "a manuscript with a painting of",
    "a manuscript with a",
    "a manuscript with",
    "a manuscript of",
    "a manuscript",
    "a close up of",
    "close up of",
    "there is",
    "a scene of",
]
# prefixes that are substrings of other prefixes must
# be checked after their parent prefix
PREFIXES.sort(key=len, reverse=True)
GENERIC = re.compile(
    r"^(a |an |the )?(black and white )?(\w+ ){0,3}"
    r"(painting|drawing|picture|image|illustration|photo|"
    r"portrait|depiction|artwork) of "
)


def log(msg: str) -> None:
    print(f"\n>>> {msg}")


def main(data_dir: Path):
    # 1) Generate BLIP captions
    images_dir = data_dir / "images"  # images are here
    if not images_dir.exists():
        raise FileNotFoundError("Images folder not found")
    meta_raw_path = data_dir / "metadata_raw.jsonl"
    if (meta_raw_path).exists():
        raise FileExistsError("metadata_raw.jsonl already exists")
    log("Generating BLIP captions")
    # set up BLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = (
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16
        )
        .to(device)
        .eval()
    )
    # run BLIP on images
    paths = sorted(images_dir.glob("*.jpg"))
    with open(meta_raw_path, "w") as meta:
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="image batches"):
            chunk = paths[i : i + BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            inp = proc(images=imgs, return_tensors="pt").to(
                device=device, dtype=torch.float16
            )
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=30)
            captions = proc.batch_decode(out, skip_special_tokens=True)
            for p, c in zip(chunk, captions, strict=True):
                meta.write(
                    json.dumps({"file_name": f"images/{p.name}", "text": c.strip()})
                    + "\n"
                )

    # 2) Clean BLIP captions
    log("Cleaning BLIP captions")
    meta_path = data_dir / "metadata.jsonl"
    with open(meta_raw_path, "r") as r, open(meta_path, "w") as out:
        for line in r:
            row = json.loads(line)
            t = row["text"].lower().strip()
            t = re.sub(r"\barafed\b|\baraffe\b|\baraf\b", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            changed = True
            while changed:
                changed = False
                for p in PREFIXES:
                    if t == p or t.startswith(p + " "):
                        t = t[len(p) :].strip()
                        changed = True
                        break  # substring prefixes can be skipped
                m = GENERIC.match(t)
                if m:
                    t = t[m.end() :].strip()
                    changed = True
            t = re.sub(r"\s+", " ", t).strip(" ,.")
            row["text"] = (
                t if len(t.split()) >= 2 else "a painting"
            )  # fallback if cleaned caption is too short
            out.write(json.dumps(row) + "\n")

    log(f"Done. Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    main(data_dir=ROOT / args.data_dir)
