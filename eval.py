import os

import torch
from fire import Fire
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_variant import DatasetVariant
from models import UNet
from schedulers.linear import LinearScheduler

T_TOTAL = 1000
NUM_SAMPLES = 50000  # 10k–50k is typical for FID
OUT_DIR = "fid_samples"


@torch.inference_mode()
def main(
    checkpoint,
    dataset="cifar10",
    batch_size=1024,
    out_dir=OUT_DIR,
    num_samples=NUM_SAMPLES,
    device="cuda",
):
    out_dir = f"{out_dir}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    variant = DatasetVariant(dataset)
    img_size = variant.img_size
    model = UNet(
        img_shape=(3, img_size, img_size), T_total=T_TOTAL, **variant.model_params
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model = torch.compile(model)

    scheduler = LinearScheduler(T=T_TOTAL, device=device)
    pbar = tqdm(total=num_samples, desc="Generating samples")
    count = 0

    while count < num_samples:
        with torch.autocast(device, dtype=torch.bfloat16):
            x = model.sample(batch_size, scheduler)
        x = x.float().cpu()
        for img in x:
            if count >= num_samples:
                break
            save_image(img, f"{out_dir}/{count:06d}.png")
            count += 1
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    Fire(main)
