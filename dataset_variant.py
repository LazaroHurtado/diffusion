import base64
from enum import Enum
from io import BytesIO

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _bytes_to_image(bytes_data):
    payload = base64.b64decode(bytes_data)
    return Image.open(BytesIO(payload)).convert("RGB")


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, label_column=None):
        self.ds = hf_dataset
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        img = item["image"]
        if self.transform:
            img = self.transform(img)

        label = -1
        if self.label_column:
            label = item[self.label_column]

        return img, label


class DatasetVariant(Enum):
    CIFAR10 = "cifar10"
    CELEB_SMALL = "celeb_small"
    CELEB = "celeb"

    @property
    def img_shape(self):
        return {
            DatasetVariant.CIFAR10: (3, 32, 32),
            DatasetVariant.CELEB_SMALL: (3, 64, 64),
            DatasetVariant.CELEB: (3, 256, 256),
        }[self]

    def _transform(self):
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _dataset(self, train):
        transform = self._transform()

        match self:
            case DatasetVariant.CIFAR10:
                hf = load_dataset("uoft-cs/cifar10", split="train").rename_column(
                    "img", "image"
                )
                return HFImageDataset(hf, transform=transform, label_column="label")
            case DatasetVariant.CELEB:
                hf = load_dataset("korexyz/celeba-hq-256x256", split="train")
                return HFImageDataset(hf, transform=transform)
            case DatasetVariant.CELEB_SMALL:
                hf = (
                    load_dataset("BeibeiLim/celeba_hq_64", split="train")
                    .cast_column("image", HFImage(decode=False))
                    .with_transform(
                        lambda x: {
                            "image": [
                                _bytes_to_image(item["path"]) for item in x["image"]
                            ]
                        }
                    )
                )
                return HFImageDataset(hf, transform=transform)
            case _:
                raise ValueError(f"Unknown variant: {self}")

    def dataloader(
        self,
        train=True,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    ):
        dataset = self._dataset(train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
