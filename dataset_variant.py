from enum import Enum
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted(Path(folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


class DatasetVariant(Enum):
    CIFAR10 = "cifar10"
    CELEB = "celeb"

    @property
    def img_size(self):
        return {
            DatasetVariant.CIFAR10: 32,
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
        }[self]

    def _transform(self):
        ops = [
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
        ]
        if self is DatasetVariant.CIFAR10:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        return transforms.Compose(ops)

    def _dataset(self, root, train):
        transform = self._transform()
        if self is DatasetVariant.CIFAR10:
            return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        if self is DatasetVariant.CELEB:
            return FlatImageDataset(Path(root) / "celeba_hq", transform=transform)
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
