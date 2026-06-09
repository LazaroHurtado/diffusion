import math

import matplotlib
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model,
        ema,
        codec,
        data_loader,
        time_scheduler,
        optimizer,
        loss_fn,
        inference_freq,
        save_freq,
        checkpoints_dir,
        images_dir,
        vlb_weight=0.001,
        device="cuda",
    ):
        self.model = model
        self.ema = ema
        self.codec = codec
        self.time_scheduler = time_scheduler
        self.device = device

        self.data_loader = data_loader

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.vlb_weight = vlb_weight
        self.inference_freq = inference_freq
        self.save_freq = save_freq
        self.checkpoints_dir = checkpoints_dir
        self.images_dir = images_dir

    def train(self, total_steps, grad_accum=1, opt_step=0, start_epoch=0):
        self.total_steps = total_steps
        self.opt_step = opt_step
        self.grad_accum = grad_accum
        self.epoch = start_epoch

        pbar = tqdm(total=self.total_steps, desc="Training")
        pbar.n = self.opt_step
        pbar.refresh()

        while self.opt_step < self.total_steps:
            self.epoch += 1
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            self._train_epoch(pbar)

            pbar.set_postfix({"epoch": self.epoch})

            if self.epoch % self.inference_freq == 0:
                self._sample_img(self.epoch)
            if self.epoch % self.save_freq == 0:
                self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()

    def _train_epoch(self, pbar):
        T_total = self.time_scheduler.T
        for idx, (x_0, y) in enumerate(self.data_loader):
            if self.opt_step >= self.total_steps:
                break

            y = (
                y.to(self.device, non_blocking=True) + 1
            )  # Reserve class label 0 for unconditional generation
            drop = torch.rand(y.size(0), device=y.device) < 0.1
            y = torch.where(drop, torch.zeros_like(y), y)
            x_0 = self.codec.encode(x_0)

            t = torch.randint(0, T_total, (x_0.size(0),), device=x_0.device)
            alpha_bar_t = self.time_scheduler.alpha_bar(t)[:, None, None, None]

            eps = torch.randn_like(x_0)
            x_t = alpha_bar_t.sqrt() * x_0 + (1.0 - alpha_bar_t).sqrt() * eps

            pred_noise, v = self.model(x_t, t, y)

            # \sum_{t=2}^{T} L_{t-1}
            # We ignore L_T, prior matching loss, and L_0, reconstruction loss
            #   - L_T does not depend on model parameters
            #   - L_0 is similar to a step in L_{1:T-1}, so we can ignore it
            l_simple = self.loss_fn(pred_noise, eps)

            # Variational lower bound comes from the improved DDPM paper
            # it measures the difference between the true posterior and the approximate posterior.
            # This makes the model's variance learnable
            l_vlb = self.vlb_term(x_0, x_t, t, v, pred_noise).mean()

            loss_value = (l_simple + self.vlb_weight * l_vlb) / self.grad_accum
            loss_value.backward()

            if (idx + 1) % self.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.ema.ema()
                self.opt_step += 1
                pbar.update(1)

    def _sample_img(self, idx):
        self.ema.model.eval()

        x = self.ema.model.sample(4, self.time_scheduler)
        x = self.codec.decode(x).cpu().numpy()
        _, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(4):
            ax = axes[i // 2, i % 2]
            ax.imshow(x[i].transpose(1, 2, 0))
            ax.axis("off")

        plt.savefig(f"{self.images_dir}/generated_samples_{idx}.png")
        plt.close()

    def _save_checkpoint(self):
        raw_model = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "model": raw_model.state_dict(),
                "ema": self.ema.state_dict(),
                "epoch": self.epoch,
                "opt_step": self.opt_step,
            },
            f"{self.checkpoints_dir}/{self.model.name}_{self.epoch}.pth",
        )

    def vlb_term(self, x_0, x_t, t, v, pred_noise):
        if v is None:
            return torch.zeros(x_0.size(0), device=x_0.device)
        v = (v + 1) / 2

        abar_t = self.time_scheduler.alpha_bar(t)[:, None, None, None]
        beta_t = self.time_scheduler.beta(t)[:, None, None, None]
        beta_tilde_t = self.time_scheduler.beta_tilde(t)[:, None, None, None]

        mu, logvar = self.time_scheduler.q_posterior(x_0, x_t, t)

        pred_x0 = (x_t - (1.0 - abar_t).sqrt() * pred_noise.detach()) / abar_t.sqrt()
        pred_mu, _ = self.time_scheduler.q_posterior(pred_x0, x_t, t)
        # Interpolation between the forward-process variance, beta_t, and the
        # true reverse-process variance, beta_tilde_t. Our model learns the interpolation weight, v
        pred_logvar = v * torch.log(beta_t) + (1 - v) * torch.log(beta_tilde_t)

        kl = self.codec.normal_kl(mu, logvar, pred_mu, pred_logvar)
        kl = kl.flatten(1).mean(1) / math.log(2.0)
        if self.codec.is_latent:
            return kl

        nll = self.codec.gaussian_nll(x_0, pred_mu, 0.5 * pred_logvar)
        nll = nll.flatten(1).mean(1) / math.log(2.0)

        return torch.where(t == 0, nll, kl)
