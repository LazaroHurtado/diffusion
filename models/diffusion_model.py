import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(self, img_shape, T_total, x0_clamp=(-1.0, 1.0), *args, **kwargs):
        super().__init__()
        assert len(img_shape) == 3, "img_shape should be (C, H, W)"
        assert img_shape[1] == img_shape[2], "img_shape H and W should be the same"

        self.img_shape = img_shape
        self.T_total = T_total
        self.x0_clamp = x0_clamp

    @torch.inference_mode()
    def sample(self, num_samples, time_scheduler, labels=None):
        device = next(self.parameters()).device
        imgs = torch.randn(num_samples, *self.img_shape, device=device)

        for t in reversed(range(self.T_total)):
            t_batch = torch.full(
                (imgs.size(0),), t, device=imgs.device, dtype=torch.long
            )
            pred_noise, v = self(imgs, t_batch, labels)

            beta_t = time_scheduler.beta(t_batch)[:, None, None, None]
            alpha_t = time_scheduler.alpha(t_batch)[:, None, None, None]
            alpha_bar_t = time_scheduler.alpha_bar(t_batch)[:, None, None, None]

            denominator = torch.clamp(1.0 - alpha_bar_t, min=1e-8).sqrt()
            pred_x0 = (imgs - denominator * pred_noise) / alpha_bar_t.sqrt()
            if self.x0_clamp is not None:
                pred_x0 = torch.clamp(pred_x0, *self.x0_clamp)

            if t == 0:
                imgs = pred_x0
                break

            alpha_bar_t_prev = time_scheduler.alpha_bar(t_batch - 1)[
                :, None, None, None
            ]

            posterior_mean = (
                beta_t * alpha_bar_t_prev.sqrt() / (1.0 - alpha_bar_t) * pred_x0
                + (1.0 - alpha_bar_t_prev) * alpha_t.sqrt() / (1.0 - alpha_bar_t) * imgs
            )

            beta_tilde_t = time_scheduler.beta_tilde(t_batch)[:, None, None, None]
            if v is not None:
                # Learned variance (Improved DDPM): interpolate in log-space between
                # beta_t and beta_tilde_t using the model's interpolation weight v.
                v = (v + 1) / 2
                log_var = v * torch.log(beta_t) + (1.0 - v) * torch.log(
                    beta_tilde_t
                )
                posterior_std = torch.exp(0.5 * log_var)
            else:
                posterior_std = beta_tilde_t.sqrt()

            z = torch.randn_like(imgs)
            imgs = posterior_mean + posterior_std * z

        return imgs
