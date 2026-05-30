import os

import matplotlib
import torch
from fire import Fire

matplotlib.use("Agg")

from dataset_variant import DatasetVariant
from models.ema import EMA
from models.unet.unet import UNet
from schedulers.linear import LinearScheduler
from trainer import Trainer


def main(
    dataset="celeb",
    data_root=".",
    batch_size=8,
    total_steps=800_000,
    T_total=1_000,
    grad_accum=8,
    start_epoch=0,
    opt_step=0,
    load_from_checkpoint=False,
    inference_freq=25,
    save_freq=100,
    device="cuda",
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
    print(f"dataset length: {len(train_loader.dataset)}")

    img_size = variant.img_size
    model = UNet(
        img_shape=(3, img_size, img_size), T_total=T_total, **variant.model_params
    ).to(device)
    if load_from_checkpoint:
        ckpt = torch.load(
            f"{checkpoints_dir}/unet_{start_epoch}.pth", map_location=device
        )
        model.load_state_dict(ckpt["model"])

    ema = EMA(model, decay=0.9999)
    if load_from_checkpoint:
        ema.load_state_dict(ckpt["ema"])

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.MSELoss()

    model = torch.compile(model)
    time_scheduler = LinearScheduler(T=T_total, device=device)
    trainer = Trainer(
        model=model,
        ema=ema,
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
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    Fire(main)
