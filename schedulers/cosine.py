import torch
import math


class CosineScheduler:
    PI_HALF = math.pi / 2.0

    def __init__(self, T, eps=0.008, device="cpu"):
        self.T = T
        self.eps = eps

        ts = torch.arange(T, dtype=torch.float64)
        f_t = self._f(ts)
        f_0 = self._f(torch.zeros(1, dtype=torch.float64))
        alpha_bar = (f_t / f_0).float()
        alpha_bar_prev = torch.cat([f_0 / f_0, alpha_bar[:-1]]).float()
        beta = torch.clamp(1.0 - alpha_bar / alpha_bar_prev, max=0.999)

        self._alpha_bar_buf = alpha_bar.to(device)  # shape (T,)
        self._beta_buf = beta.to(device)  # shape (T,)
        self._alpha_buf = (1.0 - beta).to(device)  # shape (T,)

    def _f(self, t: torch.Tensor):
        progress = t / self.T
        theta = (progress + self.eps) / (1.0 + self.eps) * self.PI_HALF
        return theta.cos().pow(2)

    def alpha_bar(self, t: torch.Tensor):
        return self._alpha_bar_buf[t]

    def alpha(self, t: torch.Tensor):
        return self._alpha_buf[t]

    def beta(self, t: torch.Tensor):
        return self._beta_buf[t]
