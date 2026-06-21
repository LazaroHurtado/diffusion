from datasets import load_dataset
from torchvision import transforms

from .base import BaseDataset


def _to_rgb(img):
    return img.convert("RGB")


class Food101Dataset(BaseDataset):
    img_shape = (3, 128, 128)
    label_column = "label"

    def _transform(self):
        return transforms.Compose(
            [
                _to_rgb,
                transforms.Resize(
                    128,
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _load(self):
        return load_dataset("ethz/food101", split="train")
