import torch.nn as nn
import torch.nn.functional as F

from models.adagn import AdaGN


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        groups=32,
        dropout=0.0,
        resample=None,
    ):
        super().__init__()
        assert resample in (None, "up", "down")
        self.resample = resample

        self.norm1 = AdaGN(in_channels, time_emb_dim, num_groups=groups)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = AdaGN(out_channels, time_emb_dim, num_groups=groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _resample(self, x):
        if self.resample == "up":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        if self.resample == "down":
            return F.avg_pool2d(x, kernel_size=2)
        return x

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb)
        h = self.act(h)
        h = self._resample(h)
        h = self.conv1(h)

        h = self.norm2(h, t_emb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        skip = self._resample(self.residual(x))
        return h + skip
