import torch

from .base_scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    def __init__(self, T, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__(T, device)

        beta = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alpha = 1.0 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        beta_tilde = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        beta_tilde[0] = beta_tilde[
            1
        ]  # beta_tilde will be used a logvar, so we overwrite t=0 to avoid log(0)

        self._set_buffers(alpha, alpha_bar, alpha_bar_prev, beta, beta_tilde)
