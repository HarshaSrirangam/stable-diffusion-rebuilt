import urllib.request
from pathlib import Path

FILES = {
    "data/weights/v1-5-pruned-emaonly.safetensors": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "data/weights/v1-5-pruned.safetensors": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors",
}

for path, url in FILES.items():
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"skipping {path}")
        continue

    print(f"downloading {path}")
    urllib.request.urlretrieve(url, path)
