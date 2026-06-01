import torch
from diffusers import AutoencoderKL


class BasicCodec:
    def __init__(self, device="cuda"):
        self.device = device

    def encode(self, x):
        return x.to(self.device, non_blocking=True)

    def decode(self, x):
        return (x + 1) / 2


class VAECodec(BasicCodec):
    def __init__(self, vae_model=None, scale=0.18215, device="cuda"):
        super().__init__(device)

        self.scale = scale
        self.device = device
        self.vae = self.build_vae(vae_model)

    def build_vae(self, vae_model):
        if vae_model is None:
            return None
        model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_model}")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        model.to(self.device)
        return model

    def encode(self, x):
        x = super().encode(x)
        with torch.no_grad():
            x = self.vae.encode(x).latent_dist.sample().mul_(self.scale)
        return x

    def decode(self, z):
        z = z / self.scale
        x = self.vae.decode(z).sample
        return super().decode(x).clamp(0, 1)
