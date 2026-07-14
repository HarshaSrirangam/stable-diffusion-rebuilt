from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
FILENAME = "v1-5-pruned-emaonly.safetensors"
DESTINATION = Path("data/weights")

target = DESTINATION / FILENAME
if target.exists():
    print(f"skipping {target}")
else:
    print(f"downloading {FILENAME}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(DESTINATION))