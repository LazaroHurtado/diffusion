from typing import Protocol

from .dit import DiT
from .ema import EMA
from .model_preset import ModelPreset
from .unet import UNet


class ModelFactory:
    MODELS = {"unet": UNet, "dit": DiT}

    @staticmethod
    def fetch_model_cls(name: str):
        model_class = ModelFactory.MODELS.get(name.lower(), None)
        if model_class is None:
            raise ValueError(f"Unknown model: {name}")
        return model_class


__all__ = ["UNet", "DiT", "EMA", "ModelPreset", "ModelFactory"]
