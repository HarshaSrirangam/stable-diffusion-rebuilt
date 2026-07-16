from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class ImageCaptionDataset(Dataset):
    """
    Image-caption dataset class. Images are resized to image_size x image_size
    squares and pixel values are scaled to [-1, 1].

    Args:
        dataset: name of dataset (e.g. naruto)
        image_size: square size images are resized to
    """
    def __init__(self, dataset: str, image_size: int = 512):
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
            self.samples = self._load_persian()
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

    def _load_persian(self):
        return load_dataset("imagefolder", data_dir="data/persian/processed", split="train")