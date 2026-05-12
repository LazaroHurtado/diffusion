import torch
import torch.nn as nn

from models.resnet import ResNetBlock
from models.time_pos_emb import PositionalEmbedding


def cat(a, b): return torch.cat([a, b], dim=1)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = self.norm(x).view(b, c, h * w).transpose(1, 2)
        h_out, _ = self.attn(h_in, h_in, h_in)
        return x + h_out.transpose(1, 2).view(b, c, h, w)


class Encoder(nn.Module):
    def __init__(self, img_size, channels, time_dim, channel_multipliers, attn_resolutions, num_resnets=2, dropout=0.1):
        super().__init__()
        self.channels = channels
        num_resolutions = len(channel_multipliers)

        in_channels = channels
        modules = []
        for i in range(num_resolutions):
            out_channels = channels * channel_multipliers[i]
            
            with_attn = img_size in attn_resolutions
            for _ in range(num_resnets):
                modules.append(ResNetBlock(in_channels, out_channels, time_dim, dropout=dropout))
                if with_attn:
                    modules.append(SelfAttention(out_channels))
                in_channels = out_channels

            if i < num_resolutions - 1:
                modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
                img_size = img_size // 2
        self.encoder = nn.ModuleList(modules)

    def forward(self, x, t_emb):
        hs = [x]
        h = x
        for module in self.encoder:
            if isinstance(module, ResNetBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            
            if isinstance(module, SelfAttention):
                hs.pop()
            hs.append(h)

        return h, hs


class Decoder(nn.Module):
    def __init__(self, img_size, channels, time_dim, channel_multipliers, attn_resolutions, num_resnets=2, dropout=0.1):
        super().__init__()
        self.channels = channels
        num_resolutions = len(channel_multipliers)

        in_channels = channels * channel_multipliers[-1]
        modules = []
        for i in reversed(range(num_resolutions)):
            out_channels = channels * channel_multipliers[i]

            with_attn = img_size in attn_resolutions
            for resnet_i in range(num_resnets+1):
                # The extra (num_resnets+1)th resnet consumes the downsampled
                # skip produced at the end of the previous (lower-i) encoder stage,
                # which has enc_channels[i-1] channels.
                if resnet_i == num_resnets and i > 0:
                    skip_in_channels = channels * channel_multipliers[i - 1]
                else:
                    skip_in_channels = channels * channel_multipliers[i]
                
                modules.append(ResNetBlock(in_channels + skip_in_channels, out_channels, time_dim, dropout=dropout))
                if with_attn:
                    modules.append(SelfAttention(out_channels))
                in_channels = out_channels

            if i > 0:
                modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
                img_size = img_size * 2
        self.decoder = nn.ModuleList(modules)

    def forward(self, h, t_emb, hs):
        for module in self.decoder:
            if isinstance(module, ResNetBlock):
                h = module(cat(h, hs.pop()), t_emb)
            else:
                h = module(h)

        return h


class UNet(nn.Module):
    def __init__(self, 
                 img_channels=3,
                 img_size=256,
                 base_channels=128,
                 time_dim=512,
                 channel_multipliers=(1, 2, 4, 8),
                 attn_resolutions=(16,),
                 num_resnets=2,
                 dropout=0.1):
        super().__init__()
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        self.encoder = Encoder(img_size, base_channels, time_dim, channel_multipliers, attn_resolutions, num_resnets, dropout=dropout)

        encoded_channels = base_channels * channel_multipliers[-1]
        self.mid1 = ResNetBlock(encoded_channels, encoded_channels, time_dim, dropout=dropout)
        self.attn_mid = SelfAttention(encoded_channels)
        self.mid2 = ResNetBlock(encoded_channels, encoded_channels, time_dim, dropout=dropout)

        img_size = img_size // (2 ** (len(channel_multipliers) - 1))
        self.decoder = Decoder(img_size, base_channels, time_dim, channel_multipliers, attn_resolutions, num_resnets, dropout=dropout)

        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.input_conv(x)

        h, hs = self.encoder(h, t_emb)

        h = self.mid1(h, t_emb)
        h = self.attn_mid(h)
        h = self.mid2(h, t_emb)

        h = self.decoder(h, t_emb, hs)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)
