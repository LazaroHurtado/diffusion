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
        self.num_classes = kwargs.get("num_classes", 0)

    @torch.inference_mode()
    def sample(
        self,
        num_samples,
        time_scheduler,
        num_steps=None,
        eta=1.0,
        labels=None,
        guidance_scale=1.0,
    ):
        device = next(self.parameters()).device
        imgs = torch.randn(num_samples, *self.img_shape, device=device)

        do_cfg = guidance_scale != 1.0 and labels is not None and self.num_classes > 0

        if num_steps is None:
            num_steps = self.T_total
        num_steps = min(num_steps, self.T_total)
        timesteps = (
            torch.linspace(0, self.T_total - 1, num_steps, device=device).round().long()
        )
        timesteps = torch.unique(timesteps).flip(0)
        last = timesteps.size(0) - 1

        for i in range(timesteps.size(0)):
            t = int(timesteps[i])
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            if do_cfg:
                pred_noise, v = self.cgf_forward(imgs, t_batch, labels, guidance_scale)
            else:
                pred_noise, v = self(imgs, t_batch, labels)

            alpha_bar_t = time_scheduler.alpha_bar(t_batch)[:, None, None, None]
            pred_x0 = self._predict_x0(imgs, pred_noise, alpha_bar_t)

            if i == last:
                imgs = pred_x0
                break

            t_prev = torch.full_like(t_batch, int(timesteps[i + 1]))
            alpha_bar_t_prev = time_scheduler.alpha_bar(t_prev)[:, None, None, None]

            if eta == 1.0 and v is not None:
                imgs = self.ddpm(imgs, pred_x0, v, alpha_bar_t, alpha_bar_t_prev)
            else:
                imgs = self.ddim(
                    imgs, pred_noise, pred_x0, alpha_bar_t, alpha_bar_t_prev, eta
                )

        return imgs

    def _predict_x0(self, x_t, pred_noise, alpha_bar_t):
        denominator = torch.clamp(1.0 - alpha_bar_t, min=1e-8).sqrt()
        pred_x0 = (x_t - denominator * pred_noise) / alpha_bar_t.sqrt()
        if self.x0_clamp is not None:
            pred_x0 = torch.clamp(pred_x0, *self.x0_clamp)
        return pred_x0

    def ddpm(self, x_t, pred_x0, v, alpha_bar_t, alpha_bar_t_prev):
        beta = (1.0 - alpha_bar_t / alpha_bar_t_prev).clamp(min=1e-20)
        beta_tilde = ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * beta).clamp(
            min=1e-20
        )

        coef_x0 = beta * alpha_bar_t_prev.sqrt() / (1.0 - alpha_bar_t)
        coef_xt = (
            (1.0 - alpha_bar_t_prev)
            * (alpha_bar_t / alpha_bar_t_prev).sqrt()
            / (1.0 - alpha_bar_t)
        )
        mean = coef_x0 * pred_x0 + coef_xt * x_t

        # Learned variance interpolates in log-space between beta and beta_tilde.
        w = (v + 1) / 2
        log_var = w * torch.log(beta) + (1.0 - w) * torch.log(beta_tilde)
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(x_t)

    def ddim(self, x_t, pred_noise, pred_x0, alpha_bar_t, alpha_bar_t_prev, eta):
        sigma = eta * (
            ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)).clamp(min=1e-8).sqrt()
            * (1.0 - alpha_bar_t / alpha_bar_t_prev).clamp(min=0.0).sqrt()
        )

        # Direction pointing to x_t, then take a step towards x_{t_prev}.
        dir_xt = (1.0 - alpha_bar_t_prev - sigma**2).clamp(min=0.0).sqrt() * pred_noise
        x_prev = alpha_bar_t_prev.sqrt() * pred_x0 + dir_xt
        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev

    def cgf_forward(self, imgs, ts, labels, guidance_scale=1.0):
        unconditioned_labels = torch.zeros_like(labels)

        x = torch.cat([imgs, imgs], dim=0)
        t = torch.cat([ts, ts], dim=0)
        y = torch.cat([labels, unconditioned_labels], dim=0)

        pred_noise, v = self(x, t, y)
        noise_cond, noise_uncond = pred_noise.chunk(2, dim=0)

        pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        if v is not None:
            v = v.chunk(2, dim=0)[0]
        return pred_noise, v
