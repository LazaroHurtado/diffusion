import os

import matplotlib
import torch
from fire import Fire

matplotlib.use("Agg")

from codec import BasicCodec, VAECodec
from dataset_variant import DatasetVariant
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


def main(
    model_name="unet",
    dataset="celeb",
    data_root=".",
    batch_size=8,
    total_steps=800_000,
    T_total=1_000,
    grad_accum=8,
    checkpoint=None,
    inference_freq=25,
    save_freq=100,
    vae_codec=None,
    device="cuda",
    **model_kwargs,
):
    variant = DatasetVariant(dataset)
    train_loader = variant.dataloader(
        root=data_root,
        train=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    checkpoints_dir = f"checkpoints/{dataset}"
    images_dir = f"images/{dataset}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"device: {device}")
    print(f"model name: {model_name}")
    print(f"dataset: {dataset}")
    print(f"dataset length: {len(train_loader.dataset)}")

    model_cls = ModelFactory.fetch_model_cls(model_name)
    model = model_cls.from_dataset(variant, T_total=T_total, **model_kwargs).to(device)
    ema = EMA(model, decay=0.9999)

    epoch, opt_step = load_from_checkpoint(model, ema, checkpoint, device)

    if vae_codec is not None:
        codec = VAECodec(vae_model=vae_codec, device=device)
    else:
        codec = BasicCodec(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.MSELoss()

    model = torch.compile(model)
    time_scheduler = CosineScheduler(T=T_total, device=device)
    trainer = Trainer(
        model=model,
        ema=ema,
        codec=codec,
        data_loader=train_loader,
        time_scheduler=time_scheduler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        inference_freq=inference_freq,
        save_freq=save_freq,
        checkpoints_dir=checkpoints_dir,
        images_dir=images_dir,
        device=device,
    )
    trainer.train(
        total_steps=total_steps,
        grad_accum=grad_accum,
        opt_step=opt_step,
        start_epoch=epoch,
    )


if __name__ == "__main__":
    Fire(main)
