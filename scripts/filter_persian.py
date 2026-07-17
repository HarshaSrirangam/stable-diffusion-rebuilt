"""
Filters raw images in data/persian/raw by comparing its CLIP embedding
to CLIP embeddings of keep/reject prompts. Likely junk images (photos of books,
photos of people, artifacts, etc.) are moved to data/persian/rejected. Runnable
on CPU.

Usage:
    uv run python scripts/filter_persian.py
"""

import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main():
    raw = Path("data/persian/raw")
    rej = raw.parent / "rejected"
    rej.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # image is likely junk if its best match is a reject prompt
    KEEP = [
        "a colorful persian miniature painting",
        "an illuminated persian manuscript page with a painting and lines of text",
        "a persian miniature painting of figures in a garden or palace",
    ]
    REJECT = [
        "a photograph of people at an event or a museum gallery",
        "a page of only handwritten arabic or persian text with no painting",
        "a photograph of an open book showing two pages",
        "a leather book cover or binding",
        "a coin or a medal",
        "a pencil sketch or a faint line drawing",
        "a black and white photograph or engraving",
        "a ceramic plate or a decorative object",
    ]
    prompts = KEEP + REJECT
    n_keep = len(KEEP)  # max index to keep

    files = sorted(raw.glob("*.jpg"))
    B = 64
    moved = 0
    for i in range(0, len(files), B):
        chunk = files[i : i + B]  # batch of images
        imgs = []  # successfully opened images
        success = []  # successfully opened image filepaths
        for img_path in chunk:  # iter over images
            try:
                imgs.append(Image.open(img_path).convert("RGB"))
                success.append(img_path)
            except Exception:
                pass
        if not imgs:
            continue
        inputs = processor(
            text=prompts, images=imgs, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            # list of len B
            # each entry: index of max logit, corresponding to prompt index
            best = (
                model(**inputs).logits_per_image.argmax(dim=1).tolist()
            )  # (B, len(prompts)) -> (B,)
        for img_path, b in zip(success, best, strict=True):
            if b >= n_keep:  # move if max logit is for a reject prompt
                shutil.move(str(img_path), str(rej / img_path.name))
                moved += 1
        print(f"{min(i+B,len(files))}/{len(files)}  moved: {moved}", end="\r")

    print(
        f"\ndone. moved {moved} images to {rej} ; kept {len(list(raw.glob('*.jpg')))}"
    )


if __name__ == "__main__":
    main()
