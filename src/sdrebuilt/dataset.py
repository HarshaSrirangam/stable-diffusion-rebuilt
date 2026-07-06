from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class ImageCaptionDataset(Dataset):
    def __init__(self, source: dict[str, str], image_size: int = 512):
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

        if source["type"] == "HuggingFace":
            self.samples = self._load_hf(source)
        else:
            raise ValueError("Unknown type")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return {
            "image": self.transform(self.samples[i]["image"]),
            "caption": self.samples[i]["text"]
        }

    def _load_hf(self, source: dict[str, str]):
        """
        source: dict with dataset type and name keys.
        """
        if source["name"] == "naruto":
            return load_dataset("lambdalabs/naruto-blip-captions", split="train")
        else:
            raise ValueError("Unknown huggingface dataset")

