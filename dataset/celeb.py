from datasets import load_dataset

from .base import BaseDataset


class CelebDataset(BaseDataset):
    img_shape = (3, 256, 256)

    def _load(self):
        return load_dataset("korexyz/celeba-hq-256x256", split="train")
