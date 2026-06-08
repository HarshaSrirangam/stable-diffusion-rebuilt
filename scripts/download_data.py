import urllib.request
from pathlib import Path

FILES = {
    "data/tokenizer/vocab.json": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json",
    "data/tokenizer/merges.txt": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt",
    "data/weights/v1-5-pruned-emaonly.ckpt": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    "data/weights/v1-5-pruned.ckpt": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt",
}

for path, url in FILES.items():
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"skipping {path}")
        continue

    print(f"downloading {path}")
    urllib.request.urlretrieve(url, path)
