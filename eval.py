import os

import torch
from fire import Fire
from torchvision.utils import save_image
from tqdm import tqdm

from codec import BasicCodec, VAECodec
from dataset_variant import DatasetVariant
from models import ModelFactory
from schedulers.linear import LinearScheduler

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
    device="cuda",
):
    out_dir = f"{out_dir}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    variant = DatasetVariant(dataset)

    model_cls = ModelFactory.fetch_model_cls(model_name)
    model = model_cls.from_dataset(variant, T_total=T_TOTAL).to(device)

    if vae_codec is not None:
        codec = VAECodec(vae_model=vae_codec, device=device)
    else:
        codec = BasicCodec(device=device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model = torch.compile(model)

    scheduler = LinearScheduler(T=T_TOTAL, device=device)
    pbar = tqdm(total=num_samples, desc="Generating samples")
    count = 0

    while count < num_samples:
        z = model.sample(batch_size, scheduler)
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
