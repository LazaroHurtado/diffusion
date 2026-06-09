from abc import abstractmethod

import torch


class BaseScheduler:
    def __init__(self, T: int, device: torch.device):
        self.T = T
        self.device = device

    @abstractmethod
    def _f(self, t: torch.Tensor):
        pass

    def _set_buffers(
        self,
        alpha: torch.Tensor,
        alpha_bar: torch.Tensor,
        alpha_bar_prev: torch.Tensor,
        beta: torch.Tensor,
        beta_tilde: torch.Tensor,
    ):
        self._alpha_buf = alpha.float().to(self.device)
        self._alpha_bar_buf = alpha_bar.float().to(self.device)
        self._alpha_bar_prev_buf = alpha_bar_prev.float().to(self.device)

        self._beta_buf = beta.float().to(self.device)
        self._beta_tilde_buf = beta_tilde.float().to(self.device)

    def alpha_bar(self, t: torch.Tensor):
        return self._alpha_bar_buf[t]

    def alpha_bar_prev(self, t: torch.Tensor):
        return self._alpha_bar_prev_buf[t]

    def alpha(self, t: torch.Tensor):
        return self._alpha_buf[t]

    def beta(self, t: torch.Tensor):
        return self._beta_buf[t]

    def beta_tilde(self, t: torch.Tensor):
        return self._beta_tilde_buf[t]

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        alpha_t = self.alpha(t).view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar(t).view(-1, 1, 1, 1)
        alpha_bar_prev_t = self.alpha_bar_prev(t).view(-1, 1, 1, 1)
        beta_t = self.beta(t).view(-1, 1, 1, 1)

        coef_x0 = beta_t * alpha_bar_prev_t.sqrt() / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev_t) * alpha_t.sqrt() / (1.0 - alpha_bar_t)

        mean = coef_x0 * x_0 + coef_xt * x_t
        logvar = self.beta_tilde(t).view(-1, 1, 1, 1).log()

        return mean, logvar
