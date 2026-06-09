import torch.nn as nn


class AdaGN(nn.Module):
    def __init__(self, num_channels, cond_dim, num_groups=32, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * num_channels),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    # AdaGN(h, y) = y_s * GroupNorm(h) + y_b
    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class AdaLN(nn.Module):
    def __init__(self, num_channels, cond_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * num_channels),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    # AdaLN(h, y) = y_s * LayerNorm(h) + y_b
    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=1)
        return h * (1 + scale[:, None, :]) + shift[:, None, :]
