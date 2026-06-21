from .base import BaseDataset
from .celeb import CelebDataset
from .celeb_small import CelebSmallDataset
from .cifar10 import CIFAR10Dataset
from .food101 import Food101Dataset
from .variant import DatasetVariant


class DatasetFactory:
    DATASETS = {
        DatasetVariant.CIFAR10: CIFAR10Dataset,
        DatasetVariant.CELEB_SMALL: CelebSmallDataset,
        DatasetVariant.CELEB: CelebDataset,
        DatasetVariant.FOOD101: Food101Dataset,
    }

    @classmethod
    def dataset_cls(cls, variant) -> type[BaseDataset]:
        variant = DatasetVariant(variant)
        dataset_cls = cls.DATASETS.get(variant)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset variant: {variant}")
        return dataset_cls

    @classmethod
    def create(cls, variant, **kwargs) -> BaseDataset:
        return cls.dataset_cls(variant)(**kwargs)

    @classmethod
    def img_shape(cls, variant):
        return cls.dataset_cls(variant).img_shape
