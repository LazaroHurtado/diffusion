import torch
import torch.nn as nn
import torch.nn.functional as F

from .adanorms import AdaGN
from .diffusion_model import DiffusionModel
from .pos_emb import SinPosEmbedding


def cat(a, b):
    return torch.cat([a, b], dim=1)


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


class SelfAttention(nn.Module):
    def __init__(self, channels, time_emb_dim, channels_per_head=64):
        super().__init__()
        assert channels % channels_per_head == 0, (
            f"channels ({channels}) must be divisible by channels_per_head ({channels_per_head})"
        )
        num_heads = channels // channels_per_head
        self.norm = AdaGN(channels, time_emb_dim)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, t_emb):
        b, c, h, w = x.shape
        h_in = self.norm(x, t_emb).view(b, c, h * w).transpose(1, 2)
        h_out, _ = self.attn(h_in, h_in, h_in)
        return x + h_out.transpose(1, 2).view(b, c, h, w)


class Encoder(nn.Module):
    def __init__(
        self,
        img_size,
        channels,
        time_dim,
        channel_multipliers,
        attn_resolutions,
        num_resnets=2,
        dropout=0.1,
    ):
        super().__init__()
        self.channels = channels
        num_resolutions = len(channel_multipliers)

        in_channels = channels
        modules = []
        for i in range(num_resolutions):
            out_channels = channels * channel_multipliers[i]

            with_attn = img_size in attn_resolutions
            for _ in range(num_resnets):
                modules.append(
                    ResNetBlock(in_channels, out_channels, time_dim, dropout=dropout)
                )
                if with_attn:
                    modules.append(SelfAttention(out_channels, time_dim))
                in_channels = out_channels

            if i < num_resolutions - 1:
                down = ResNetBlock(
                    out_channels,
                    out_channels,
                    time_dim,
                    dropout=dropout,
                    resample="down",
                )
                modules.append(down)
                img_size = img_size // 2
        self.encoder = nn.ModuleList(modules)

    def forward(self, x, t_emb):
        hs = [x]
        h = x
        for module in self.encoder:
            if isinstance(module, (ResNetBlock, SelfAttention)):
                h = module(h, t_emb)

            if isinstance(module, SelfAttention):
                hs.pop()
            hs.append(h)

        return h, hs


class Decoder(nn.Module):
    def __init__(
        self,
        img_size,
        channels,
        time_dim,
        channel_multipliers,
        attn_resolutions,
        num_resnets=2,
        dropout=0.1,
    ):
        super().__init__()
        self.channels = channels
        num_resolutions = len(channel_multipliers)

        in_channels = channels * channel_multipliers[-1]
        modules = []
        for i in reversed(range(num_resolutions)):
            out_channels = channels * channel_multipliers[i]

            with_attn = img_size in attn_resolutions
            for resnet_i in range(num_resnets + 1):
                # The extra (num_resnets+1)th resnet consumes the downsampled
                # skip produced at the end of the previous (lower-i) encoder stage,
                # which has enc_channels[i-1] channels.
                if resnet_i == num_resnets and i > 0:
                    skip_in_channels = channels * channel_multipliers[i - 1]
                else:
                    skip_in_channels = channels * channel_multipliers[i]

                modules.append(
                    ResNetBlock(
                        in_channels + skip_in_channels,
                        out_channels,
                        time_dim,
                        dropout=dropout,
                    )
                )
                if with_attn:
                    modules.append(SelfAttention(out_channels, time_dim))
                in_channels = out_channels

            if i > 0:
                up = ResNetBlock(
                    in_channels, in_channels, time_dim, dropout=dropout, resample="up"
                )
                modules.append(up)
                img_size = img_size * 2
        self.decoder = nn.ModuleList(modules)

    def forward(self, h, t_emb, hs):
        for module in self.decoder:
            if isinstance(module, ResNetBlock) and not getattr(
                module, "resample", None
            ):
                h = module(cat(h, hs.pop()), t_emb)
            else:
                h = module(h, t_emb)

        return h


class UNet(DiffusionModel):
    def __init__(
        self,
        img_shape=(3, 256, 256),
        base_channels=128,
        time_dim=512,
        channel_multipliers=(1, 2, 4, 8),
        attn_resolutions=(16,),
        num_resnets=2,
        dropout=0.1,
        *args,
        **kwargs,
    ):
        super().__init__(img_shape, *args, **kwargs)
        self.img_channels = img_shape[0]
        self.img_size = img_shape[1]
        self.time_embedding = nn.Sequential(
            SinPosEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(
            self.img_channels, base_channels, kernel_size=3, padding=1
        )

        self.encoder = Encoder(
            self.img_size,
            base_channels,
            time_dim,
            channel_multipliers,
            attn_resolutions,
            num_resnets,
            dropout=dropout,
        )

        encoded_channels = base_channels * channel_multipliers[-1]
        self.mid1 = ResNetBlock(
            encoded_channels, encoded_channels, time_dim, dropout=dropout
        )
        self.attn_mid = SelfAttention(encoded_channels, time_dim)
        self.mid2 = ResNetBlock(
            encoded_channels, encoded_channels, time_dim, dropout=dropout
        )

        decoder_img_size = self.img_size // (2 ** (len(channel_multipliers) - 1))
        self.decoder = Decoder(
            decoder_img_size,
            base_channels,
            time_dim,
            channel_multipliers,
            attn_resolutions,
            num_resnets,
            dropout=dropout,
        )

        self.out_norm = AdaGN(base_channels, time_dim)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            base_channels, self.img_channels, kernel_size=3, padding=1
        )

    # UNet implementation is classifier free, so we ignore any class label that is being passed.
    def forward(self, x, t, *args, **kwargs):
        t_emb = self.time_embedding(t)
        h = self.input_conv(x)

        h, hs = self.encoder(h, t_emb)

        h = self.mid1(h, t_emb)
        h = self.attn_mid(h, t_emb)
        h = self.mid2(h, t_emb)

        h = self.decoder(h, t_emb, hs)

        h = self.out_norm(h, t_emb)
        h = self.out_act(h)

        return self.out_conv(h), None
