from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from .model.autoencoder import Autoencoder
from .model.clip import CLIP
from .convert_weights import load_vae, load_clip


class ImageCaptionDataset(Dataset):
    """
    Image-caption dataset class. Images are resized to image_size x image_size
    squares and pixel values are scaled to [-1, 1].

    Args:
        dataset: name of dataset (e.g. naruto)
        split: training or eval split, used only for persian
        image_size: square size images are resized to
    """
    def __init__(
            self,
            dataset: str,
            split: str = "train",
            image_size: int = 512
        ):
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")), # 3 channels
            transforms.Resize(image_size), # shorter side -> 512
            transforms.CenterCrop(image_size), # -> 512x512, no distortion
            transforms.ToTensor(), # PIL -> (3, H, W) float tensor in [0, 1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ) # [0, 1] -> [-1, 1] for VAE/UNet
        ])

        if dataset == "naruto":
            self.samples = self._load_naruto()
        elif dataset == "persian":
            self.samples = self._load_persian(split=split)
        else:
            raise ValueError("Unknown type")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        """
        Naruto samples are accessed via samples[i]["image"] and 
        samples[i]["text"], so other custom dataset(s) was built
        to be indexed the same way.
        """
        return {
            "image": self.transform(self.samples[i]["image"]),
            "caption": self.samples[i]["text"]
        }

    def _load_naruto(self):
        return load_dataset("lambdalabs/naruto-blip-captions", split="train")

    def _load_persian(self, split: str):
        return load_dataset("imagefolder", data_dir=f"data/persian/{split}", split="train")


@torch.no_grad()
def precompute(
        pretrained_path: Path,
        dataset: str,
        split: str,
        batch_size: int,
        device: torch.device,
        cache_path: Path
    ) -> None:
    """
    Precomputes and saves images->latents and captions->clip embeddings to disk.
    """
    # build frozen encoders
    vae = Autoencoder().eval().requires_grad_(False)
    load_vae(path=pretrained_path, vae=vae)
    vae.to(device)
    clip = CLIP().eval().requires_grad_(False)
    load_clip(path=pretrained_path, clip=clip)
    clip.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # build raw dataset
    ds = ImageCaptionDataset(
        dataset=dataset,
        split=split,
        image_size=512,
    )
    loader = DataLoader(ds, batch_size=batch_size)

    # encode data
    latents_list, context_list = [], []
    pbar = tqdm(loader, desc="Encoding image/caption batches", colour="blue")
    for batch in pbar:
        images = batch["image"].to(device)
        captions = batch["caption"]
        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device=device, dtype=torch.long)
        encoder_noise = torch.randn((images.size(0), 4, 64, 64), device=device)
        latents_list.append(vae.encode(images, encoder_noise).cpu())
        context_list.append(clip(tokens).cpu())
    vae.to("cpu")
    clip.to("cpu")
    torch.cuda.empty_cache()

    # save embeddings to disk
    latents = torch.cat(latents_list)  # (B, 4, 64, 64)
    context = torch.cat(context_list)  # (B, 77, 768)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "context": context}, cache_path)