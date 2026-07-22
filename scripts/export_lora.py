"""
Turn a run's last checkpoint + config into a single weight file.
"""

import argparse
from pathlib import Path
import torch
import yaml
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

def main(run: str, out: str):
    run_dir = ROOT / "runs" / run
    config = yaml.safe_load(open(run_dir / "training_config.yaml"))
    ckpts = (run_dir / "checkpoints").glob("checkpoint-*.pt")
    last = max(ckpts, key=lambda p: int(p.stem.split("-")[1]))
    state = torch.load(last, map_location="cpu")
    save_file(
        state,                                  
        ROOT / out,
        metadata={"r": str(config["r"]), "alpha": str(config["alpha"]),
                "targets": ",".join(config["targets"]["layers"])},
    )
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--out", type=str, default="persian_lora.safetensors")
    args = parser.parse_args()
    main(
        run=args.run, out=args.out
    )
