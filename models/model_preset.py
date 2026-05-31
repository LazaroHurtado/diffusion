from typing import Protocol

from dataset_variant import DatasetVariant


class ModelPreset(Protocol):
    def name(self) -> str: ...

    @classmethod
    def with_preset(cls, name: str, **kwargs) -> "ModelPreset": ...

    @classmethod
    def from_dataset(cls, dataset: DatasetVariant, **kwargs) -> "ModelPreset": ...
