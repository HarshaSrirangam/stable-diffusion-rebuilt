from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class ImageCaptionDataset(Dataset):
    """
    Image-caption dataset class.

    Loads image-caption pairs from the given source. Indexing returns a dict:
    {"image": Tensor(3, image_size, image_size) in [-1, 1], "caption": str}

    Args:
        dataset_name: which dataset to load (e.g. "naruto")
        dataset_type: source type (e.g. "HuggingFace)
        image_size: square size images are resized to
    """
    def __init__(self, dataset_name: str, dataset_type: str, image_size: int = 512):
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

        if dataset_type == "HuggingFace":
            self.samples = self._load_hf(dataset_name)
        else:
            raise ValueError("Unknown type")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return {
            "image": self.transform(self.samples[i]["image"]),
            "caption": self.samples[i]["text"]
        }

    def _load_hf(self, dataset_name: str):
        if dataset_name == "naruto":
            return load_dataset("lambdalabs/naruto-blip-captions", split="train")
        else:
            raise ValueError("Unknown HuggingFace dataset")

