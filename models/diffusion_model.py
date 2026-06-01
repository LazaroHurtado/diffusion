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
            pred_noise, _ = self(imgs, t_batch, labels)

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

            posterior_variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
            z = torch.randn_like(imgs)
            imgs = posterior_mean + posterior_variance.sqrt() * z

        return imgs
