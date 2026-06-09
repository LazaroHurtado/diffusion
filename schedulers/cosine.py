import math

import torch

from .base_scheduler import BaseScheduler


class CosineScheduler(BaseScheduler):
    def __init__(self, T, eps=0.008, device="cpu"):
        super().__init__(T, device)
        self.eps = eps

        ts = torch.arange(T + 1, dtype=torch.float64)
        f_t = self._f(ts)
        f_0 = f_t[0]

        alpha_bar_full = (f_t / f_0).float()
        beta = torch.clamp(1.0 - alpha_bar_full[1:] / alpha_bar_full[:-1], max=0.999)
        alpha = 1.0 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        beta_tilde = (beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).clamp(
            min=1e-20
        )
        beta_tilde[0] = beta_tilde[
            1
        ]  # beta_tilde will be used a logvar, so we overwrite t=0 to avoid log(0)

        self._set_buffers(alpha, alpha_bar, alpha_bar_prev, beta, beta_tilde)

    def _f(self, t: torch.Tensor):
        progress = t / self.T
        theta = (progress + self.eps) / (1.0 + self.eps) * (math.pi / 2.0)
        return theta.cos().pow(2)
