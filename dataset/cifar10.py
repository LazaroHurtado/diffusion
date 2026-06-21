from datasets import load_dataset

from .base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    img_shape = (3, 32, 32)
    label_column = "label"

    def _load(self):
        return load_dataset("uoft-cs/cifar10", split="train").rename_column(
            "img", "image"
        )
