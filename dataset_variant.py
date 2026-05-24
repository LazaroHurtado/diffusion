from enum import Enum
from pathlib import Path
import base64

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset, Image as HFImage
from io import BytesIO


def _bytes_to_image(bytes_data):
    payload = base64.b64decode(bytes_data)
    return Image.open(BytesIO(payload)).convert("RGB")


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        if self.transform:
            img = self.transform(img)
        return img, 0


class DatasetVariant(Enum):
    CIFAR10 = "cifar10"
    CELEB_SMALL = "celeb_small"
    CELEB = "celeb"

    @property
    def img_size(self):
        return {
            DatasetVariant.CIFAR10: 32,
            DatasetVariant.CELEB_SMALL: 64,
            DatasetVariant.CELEB: 256,
        }[self]
    
    @property
    def model_params(self):
        return {
            DatasetVariant.CIFAR10: {
                "base_channels": 128,
                "time_dim": 512,
                "channel_multipliers": (1, 2, 2, 2),
                "attn_resolutions": (16,),
                "num_resnets": 2,
                "dropout": 0.1,
            },
            DatasetVariant.CELEB: {
                "base_channels": 128,
                "time_dim": 512,
                "channel_multipliers": (1, 1, 2, 2, 4, 4),
                "attn_resolutions": (16,),
                "num_resnets": 2,
                "dropout": 0.1,
            },
            DatasetVariant.CELEB_SMALL: {
                "base_channels": 128,
                "time_dim": 512,
                "channel_multipliers": (1, 2, 2, 2),
                "attn_resolutions": (16,),
                "num_resnets": 2,
                "dropout": 0.1,
            }
        }[self]

    def _transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def _dataset(self, root, train):
        transform = self._transform()
        if self is DatasetVariant.CIFAR10:
            return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        
        if self is DatasetVariant.CELEB:
            hf = load_dataset("korexyz/celeba-hq-256x256", split="train")
            return HFImageDataset(hf, transform=transform)
        
        if self is DatasetVariant.CELEB_SMALL:
            hf = load_dataset("BeibeiLim/celeba_hq_64", split="train") \
                    .cast_column("image", HFImage(decode=False)) \
                    .with_transform(
                        lambda x: {"image": [_bytes_to_image(item["path"]) for item in x["image"]]}
                    )
            return HFImageDataset(hf, transform=transform)
        
        raise ValueError(f"Unknown variant: {self}")

    def dataloader(
        self,
        root="./data",
        train=True,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    ):
        dataset = self._dataset(root, train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
