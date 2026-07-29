# Stable Diffusion 1.5 reimplemented from scratch in PyTorch and LoRA finetuned on Persian miniature paintings

<p align="center">
  <img src="assets/garden.png" width="32%" />
  <img src="assets/feast.png"   width="32%" />
  <img src="assets/battle.png"  width="32%" />
</p>

A from-scratch reimplementation of Stable Diffusion 1.5 in PyTorch: the VAE, CLIP text
encoder, UNet, and samplers are all built by hand (no `diffusers`). I did not pretrain 
the base model from scratch; I instead loaded the checkpoint. I then LoRA-finetuned
the base model into a Persian-miniature style adapter.

<!-- TODO(you): 2–4 sentences in your own words. What's notable, why you built every
component from scratch instead of using diffusers, and what you learned / what the repo
is for (portfolio / understanding SD internals). This is the paragraph people read first
after the images, so make it yours. -->

## Components

- From-scratch base model: CLIP text encoder, VAE, and UNet (self/cross attention + FFN)
- Samplers: DDPM, DDIM implemented from scratch, Euler sampler is in progress
- LoRA: low rank adapters with training code
- txt2img and img2img inference.
- Persian-miniature LoRA: trained on a scraped and cleaned Wikimedia dataset; weights and dataset published to Hugging Face Hub.

<!-- TODO(you): trim/rewrite these bullets to match how you'd describe it. -->

## Results

<!-- TODO(you): 1–2 sentences — what you set out to test and the headline result (the adapter
shifts base SD1.5 toward the Persian-miniature distribution). First person reads well here. -->

### How I ran experiments

I finetuned LoRA adapters varying three degrees of freedom between experiments: rank (`r`) (adapter capacity),
alpha (the `alpha / r` scaling in the LoRA update), and target layers (which layers in the base model get a LoRA layer attached, e.g. q/k/v/out proj self-attention layers). Each run is scored by `scripts/evaluate_lora.py`, which generates images 
on a fixed prompt set and computes the three metrics below:

| Run | r | α | Target layers | Val-loss ratio | Prompt adh. | Style adh. |
|---|---|---|---|---|---|---|
| self | 16 | 8 | self-attn | 0.9924 | 0.313 | 0.680 |
| cross | 16 | 8 | cross | 0.9948 | 0.290 | 0.737 |
| self+cross | 16 | 8 | self + cross | 0.9941 | 0.289 | 0.725 |
| self+cross | 16 | 16 | self + cross | 0.9984 | 0.290 | 0.710 |
| **self+cross+FFN** | **16** | **8** | **self + cross + FFN** | **0.9933** | **0.303** | **0.747** |
| self | 32 | 16 | self-attn | 0.9934 | 0.304 | 0.734 |
| cross | 32 | 16 | cross | 0.9961 | 0.278 | 0.719 |
| self+cross | 32 | 16 | self + cross | 0.9970 | 0.278 | 0.757 |
| self+cross | 32 | 32 | self + cross | 1.0072 | 0.267 | 0.630 |
| self+cross+FFN | 32 | 16 | self + cross + FFN | 0.9981 | 0.284 | 0.738 |

<!-- TODO(you): one row per run in your sweep — fill r / α / targets and the numbers from each
runs/<run>/eval/metrics.json. Bold the winning row. Add/remove rows to match your actual runs.
Metric definitions:
  - Val-loss ratio  = LoRA / base noise-prediction MSE on held-out eval images (< 1.0 is better)
  - Prompt adherence = generated image ↔ prompt CLIP similarity (photo-biased, lower for stylized)
  - Style adherence  = generated image ↔ eval-set style-centroid CLIP similarity (higher = more in-style)
The CLIP metrics are comparative across runs, not absolute. -->

### Winning configuration

<!-- TODO(you): state the winner (r = 16, alpha = 8, targets = self + cross + FFN) and why it won —
best style adherence without wrecking prompt adherence, lowest val-loss ratio, best eval grids.
Optional "what I learned" beat: the low-alpha divergence I hit (Adam pushing raw weights large in
bf16) and the gradient-clipping fix.
NOTE: committed configs/lora.yaml is r16 / alpha16 / self+cross (no FFN) — NOT the winning run.
Reconcile these before documenting. -->

**Base vs. LoRA** (same prompt and seed):

<p align="center">
  <img src="assets/base_vs_lora.png" width="60%" />
</p>

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. clone and install
git clone https://github.com/HarshaSrirangam/stable-diffusion-rebuilt.git
cd stable-diffusion-rebuilt
uv sync

# 2. download the base SD1.5 checkpoint + Persian LoRA weights (into data/weights/)
uv run python scripts/download_data.py

# 3. edit configs/inference.yaml (prompt, sampler, seed, lora_path, ...), then generate
uv run python scripts/generate.py
```

Generated images are written to `outputs/out<N>/` along with a snapshot of the config used.
Set `lora_path: null` in `configs/inference.yaml` to run the base model instead of the LoRA,
or point `input_image` at a file for img2img. See `notebooks/inference_demo.ipynb` for a
notebook walkthrough.

## How it works

<!-- TODO(you): a short tour of the pipeline in your own words — how text → CLIP embedding →
UNet denoising in latent space → VAE decode fits together, and where LoRA is injected. This is
where the "from scratch" work pays off, so explain what you actually built. A small architecture
diagram helps if you want one. Good place to mention the divergence/gradient-clipping story too. -->

## Reproducing the LoRA

```bash
# build the dataset (scrape → filter → caption), or pull the published parquet dataset
#   see notebooks/build_dataset.ipynb  (GPU recommended for captioning)

# train a LoRA run (reads configs/lora.yaml, writes to runs/<name>/)
uv run python scripts/train_lora.py

# evaluate a run (metrics + eval grid → runs/<run>/eval/)
uv run python scripts/evaluate_lora.py --run <run_name>

# export a run's checkpoint into a portable, self-describing .safetensors
uv run python scripts/export_lora.py --run <run_name> --out persian_lora.safetensors
```

<!-- TODO(you): note the winning config (r / alpha / target layers) and anything you'd want a
reader to know about the sweep. -->

The dataset-authoring scripts (`scrape.py`, `filter_persian.py`, `caption.py`) live in `scripts/`
and are orchestrated by `notebooks/build_dataset.ipynb`.

## Repository structure

```
stable-diffusion-rebuilt/
├── src/sdrebuilt/
│   ├── model/             # from-scratch VAE, CLIP text encoder, UNet
│   ├── samplers/          # DDPM, DDIM (Euler in progress)
│   ├── lora/              # LoRA layers + inject/enable/disable/merge utility functions
│   ├── inference.py       # txt2img / img2img pipeline
│   ├── trainer.py         # LoRA training loop
│   ├── dataset.py         # dataset + latent/text-embedding precompute
│   └── convert_weights.py # load SD1.5 .safetensors into the from-scratch modules
├── scripts/
│   ├── scrape.py          # scrape Persian miniatures from Wikimedia Commons
│   ├── filter_persian.py  # CLIP + saturation junk filter
│   ├── caption.py         # BLIP captioning + cleanup
│   ├── train_lora.py      # train a LoRA run from configs/lora.yaml
│   ├── evaluate_lora.py   # metrics + eval grid for a run
│   ├── export_lora.py     # bundle a run's checkpoint into a portable .safetensors
│   ├── generate.py        # config-driven inference (configs/inference.yaml)
│   └── download_data.py   # fetch base SD1.5 + Persian LoRA weights from HF
├── configs/               # inference.yaml, lora.yaml, samplers.yaml
├── notebooks/             # build_dataset, train, inference_demo, lora_inference
└── assets/                # sample generations
```

## Data and weights

- **LoRA weights:** [`HarshaSrirangam/persian-miniature-lora`](https://huggingface.co/HarshaSrirangam/persian-miniature-lora)
- **Dataset:** [`HarshaSrirangam/persian-miniatures`](https://huggingface.co/datasets/HarshaSrirangam/persian-miniatures)
- **Base model:** [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (`v1-5-pruned-emaonly.safetensors`)

Source images were scraped from [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Persian_miniatures)
(~6.8k scraped → CLIP + saturation filter → ~5.2k kept → captioned and published as parquet).
<!-- TODO(you): note the image license / attribution for the dataset. -->

## Acknowledgements

<!-- TODO(you): references / credits, e.g.:
  - Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models (SD)
  - Hu et al., LoRA: Low-Rank Adaptation of Large Language Models
  - Ho et al., DDPM; Song et al., DDIM
  - any tutorials that guided you
-->

## License

<!-- TODO(you): add a license (MIT is common for repos like this) and a LICENSE file. -->