import torch


class LinearScheduler:
    def __init__(self, T, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T

        beta = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        self._beta_buf = beta.float().to(device)
        self._alpha_buf = alpha.float().to(device)
        self._alpha_bar_buf = alpha_bar.float().to(device)
        self._alpha_bar_prev_buf = alpha_bar_prev.float().to(device)

    def alpha_bar(self, t: torch.Tensor):
        return self._alpha_bar_buf[t]

    def alpha(self, t: torch.Tensor):
        return self._alpha_buf[t]

    def beta(self, t: torch.Tensor):
        return self._beta_buf[t]
