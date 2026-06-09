import numpy as np
import torch
from diffusers import AutoencoderKL


class BasicCodec:
    def __init__(self, is_latent=False, device="cuda"):
        self.is_latent = is_latent
        self.device = device

    def encode(self, x):
        return x.to(self.device, non_blocking=True)

    def decode(self, x):
        return (x + 1) / 2

    def normal_kl(self, mu1, logvar1, mu2, logvar2):
        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + (mu1 - mu2) ** 2 * torch.exp(-logvar2)
        )

    def std_normal_cdf(self, x):
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def gaussian_nll(self, x, mus, logvars):
        c = x - mus
        inv = torch.exp(-logvars)

        cdf_plus = self.std_normal_cdf((c + 1 / 255) * inv)
        cdf_min = self.std_normal_cdf((c - 1 / 255) * inv)

        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_cdf_min = torch.log((1 - cdf_min).clamp(min=1e-12))

        cdf_delta = (cdf_plus - cdf_min).clamp(min=1e-12)
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_cdf_min, torch.log(cdf_delta)),
        )
        return -log_probs


class VAECodec(BasicCodec):
    def __init__(self, vae_model=None, scale=0.18215, device="cuda"):
        super().__init__(is_latent=True, device=device)

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
