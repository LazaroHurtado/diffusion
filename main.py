import os

import matplotlib
import torch
from fire import Fire

matplotlib.use("Agg")

from codec import BasicCodec, VAECodec
from config import Config
from dataset import DatasetFactory
from models import EMA, ModelFactory
from schedulers import CosineScheduler
from trainer import Trainer


def load_from_checkpoint(model, ema, checkpoint, device):
    if checkpoint is None or not checkpoint.endswith(".pth"):
        return 0, 0

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])

    epoch = ckpt["epoch"]
    opt_step = ckpt["opt_step"]
    return epoch, opt_step


def main(config_file="config.yml"):
    cfg = Config.from_yaml(config_file)
    train_cfg = cfg.trainer
    ds_cfg = cfg.dataset
    model_cfg = cfg.model

    variant = ds_cfg.variant
    train_dataset = DatasetFactory.create(variant, train=True)
    train_loader = train_dataset.to_dataloader(
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=ds_cfg.prefetch_factor,
    )

    checkpoints_dir = f"checkpoints/{variant.value}"
    images_dir = f"images/{variant.value}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    device = train_cfg.device
    print(f"device: {device}")
    print(f"model name: {model_cfg.name}")
    print(f"dataset: {variant.value}")
    print(f"dataset length: {len(train_loader.dataset)}")

    model_cls = ModelFactory.fetch_model_cls(model_cfg.name)
    model = model_cls.from_dataset(
        variant, T_total=train_cfg.T_total, **model_cfg.params
    ).to(device)
    ema = EMA(model, decay=0.9999)

    epoch, opt_step = load_from_checkpoint(model, ema, train_cfg.checkpoint, device)

    if train_cfg.codec is not None:
        codec = VAECodec(vae_model=train_cfg.codec.value, device=device)
    else:
        codec = BasicCodec(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.MSELoss()

    model = torch.compile(model)
    time_scheduler = CosineScheduler(T=train_cfg.T_total, device=device)
    trainer = Trainer(
        model=model,
        ema=ema,
        codec=codec,
        data_loader=train_loader,
        time_scheduler=time_scheduler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        inference_freq=train_cfg.inference_frequency,
        save_freq=train_cfg.save_frequency,
        checkpoints_dir=checkpoints_dir,
        images_dir=images_dir,
        guidance_scale=train_cfg.guidance_scale,
        min_snr_gamma=train_cfg.min_snr_gamma,
        device=device,
    )
    trainer.train(
        total_steps=train_cfg.total_steps,
        grad_accum=train_cfg.grad_accum,
        opt_step=opt_step,
        start_epoch=epoch,
    )


if __name__ == "__main__":
    Fire(main)
