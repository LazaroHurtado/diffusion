import os

import torch
from fire import Fire
from torchvision.utils import save_image
from tqdm import tqdm

from codec import BasicCodec, VAECodec
from dataset import DatasetVariant
from models import ModelFactory
from schedulers import CosineScheduler

T_TOTAL = 1000
NUM_SAMPLES = 50000  # 10k–50k is typical for FID
OUT_DIR = "fid_samples"


@torch.inference_mode()
def main(
    checkpoint,
    model_name="unet",
    vae_codec=None,
    dataset="cifar10",
    batch_size=1024,
    out_dir=OUT_DIR,
    num_samples=NUM_SAMPLES,
    num_steps=None,
    eta=1.0,
    device="cuda",
    **model_kwargs,
):
    out_dir = f"{out_dir}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    variant = DatasetVariant(dataset)

    if vae_codec is not None:
        codec = VAECodec(vae_model=vae_codec, device=device)
        model_kwargs.setdefault("x0_clamp", None)
    else:
        codec = BasicCodec(device=device)

    model_cls = ModelFactory.fetch_model_cls(model_name)
    model = model_cls.from_dataset(variant, T_total=T_TOTAL, **model_kwargs).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["ema"]["model"]
    model.load_state_dict(state)
    model.eval()
    model = torch.compile(model)

    scheduler = CosineScheduler(T=T_TOTAL, device=device)
    pbar = tqdm(total=num_samples, desc="Generating samples")
    count = 0

    while count < num_samples:
        n = min(batch_size, num_samples - count)
        z = model.sample(n, scheduler, num_steps=num_steps, eta=eta)
        x = codec.decode(z.float()).cpu()
        for img in x:
            if count >= num_samples:
                break
            save_image(img, f"{out_dir}/{count:06d}.png")
            count += 1
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    Fire(main)
