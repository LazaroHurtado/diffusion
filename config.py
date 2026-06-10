from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel

from dataset_variant import DatasetVariant


class CodecEnum(str, Enum):
    MSE = "mse"
    MSE_ORIGINAL = "mse_original"
    EMA = "ema"
    EMA_ORIGINAL = "ema_original"


class TrainerConfig(BaseModel):
    total_steps: int = 800_000
    T_total: int = 1000

    codec: Optional[CodecEnum] = None

    grad_accum: int = 2
    device: str = "cuda"
    checkpoint: Optional[str] = None
    inference_frequency: int = 25
    save_frequency: int = 100


class DatasetConfig(BaseModel):
    variant: DatasetVariant = DatasetVariant.CELEB
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 8


class ModelConfig(BaseModel):
    name: str = "unet"
    params: dict = {}


class Config(BaseModel):
    trainer: TrainerConfig
    model: ModelConfig
    dataset: DatasetConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
