import torch
import torch.nn as nn

from dataset import DatasetFactory, DatasetVariant

from .adanorms import AdaLN
from .diffusion_model import DiffusionModel
from .model_preset import ModelPreset
from .pos_emb import PatchSinPosEmbedding, SinPosEmbedding


class BaseDitBlock(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.conditioning = nn.Linear(embed_dim, dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )


class DiTBlock(BaseDitBlock):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__(dim, embed_dim, num_heads, mlp_ratio)
        self.norm1 = AdaLN(dim, dim)
        self.scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.norm2 = AdaLN(dim, dim)

        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)

    def forward(self, x, c):
        c = self.conditioning(c)
        scale1, scale2 = self.scale(c).unsqueeze(1).chunk(2, dim=-1)

        # Self-attention
        normed_x = self.norm1(x, c)
        attn_out = self.self_attn(normed_x, normed_x, normed_x)[0]
        x = x + (attn_out * scale1)

        # MLP
        normed_x = self.norm2(x, c)
        mlp_out = self.mlp(normed_x)
        x = x + (mlp_out * scale2)

        return x


class DiTCrossAttentionBlock(BaseDitBlock):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__(dim, embed_dim, num_heads, mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, c):
        c = self.conditioning(c)

        # Self-attention
        normed_x = self.norm1(x)
        x = x + self.self_attn(normed_x, normed_x, normed_x)[0]

        # Cross-attention
        normed_x = self.norm2(x)
        x = x + self.cross_attn(normed_x, c, c)[0]

        # MLP
        normed_x = self.norm3(x)
        x = x + self.mlp(normed_x)

        return x


class DiTInContextBlock(BaseDitBlock):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__(dim, embed_dim, num_heads, mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, c):
        c = self.conditioning(c)

        # Concatenate the time embedding with the input
        x = torch.cat([x, c], dim=-1)

        # Self-attention
        normed_x = self.norm1(x)
        x = x + self.self_attn(normed_x, normed_x, normed_x)[0]

        # MLP
        normed_x = self.norm2(x)
        x = x + self.mlp(normed_x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, grid, patch_size, in_chans, embed_dim):
        super().__init__()

        self.projection = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = PatchSinPosEmbedding(grid, embed_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [batch_size, grid_h * grid_w, embed_dim]
        x = self.pos_embed(x)
        return x


class DiT(DiffusionModel, ModelPreset):
    PRESETS = {
        "dit-s": {
            "num_layers": 12,
            "embed_dim": 384,
            "num_heads": 6,
        },
        "dit-b": {
            "num_layers": 12,
            "embed_dim": 768,
            "num_heads": 12,
        },
        "dit-l": {
            "num_layers": 24,
            "embed_dim": 1024,
            "num_heads": 16,
        },
        "dit-xl": {
            "num_layers": 28,
            "embed_dim": 1152,
            "num_heads": 16,
        },
    }

    def __init__(
        self,
        img_shape,
        patch_size,
        embed_dim,
        num_layers,
        num_heads,
        num_classes=0,
        mlp_ratio=4.0,
        block_type=DiTBlock,
        name="dit",
        *args,
        **kwargs,
    ):
        super().__init__(img_shape=img_shape, *args, **kwargs)
        assert len(img_shape) == 3

        self.name = name
        self.img_size = img_shape[1:]
        self.in_chans = img_shape[0]
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.grid = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)

        self.patch_embed = PatchEmbedding(
            self.grid, patch_size, self.in_chans, embed_dim
        )
        self.time_pos_embed = SinPosEmbedding(embed_dim)
        self.label_embed = (
            nn.Embedding(num_classes + 1, embed_dim) if num_classes > 0 else None
        )

        self.dit_blocks = nn.ModuleList(
            [
                block_type(embed_dim, embed_dim, num_heads, mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.norm = AdaLN(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, patch_size * patch_size * self.in_chans * 2)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.in_chans
        grid_h, grid_w = self.grid

        # The goal is to convert the patchified image back to the original image
        # but we want both the mean and variance for every pixel, hence the 2*c

        # x -> [batch_size, grid_h * grid_w, p * p * 2 * c]
        x = x.reshape(
            -1, grid_h, grid_w, p, p, 2 * c
        )  # [batch_size, grid_h, grid_w, patch_size, patch_size, 2* in_chans]
        x = (
            x.permute(0, 5, 1, 3, 2, 4).contiguous()
        )  # [batch_size, 2 * in_chans, grid_h, patch_size, grid_w, patch_size]
        x = x.view(
            -1, 2 * c, grid_h * p, grid_w * p
        )  # [batch_size, 2 * in_chans, grid_h * patch_size, grid_w * patch_size]
        return x

    def forward(self, x, t, y=None):
        x = self.patch_embed(x)
        c = self.time_pos_embed(t)

        if self.label_embed is not None:
            if y is None:
                y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            c = c + self.label_embed(y)

        for block in self.dit_blocks:
            x = block(x, c)

        x = self.out(self.norm(x, c))

        x = self.unpatchify(x)
        mean, v = x.chunk(2, dim=1)

        return mean, v

    @classmethod
    def with_preset(cls, preset_name: str, **kwargs) -> "DiT":
        preset = cls.PRESETS.get(preset_name.lower(), None)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")
        return cls(name=preset_name, **preset, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: DatasetVariant, **kwargs) -> "DiT":
        img_shape = DatasetFactory.img_shape(dataset)
        kwargs["img_shape"] = kwargs.get("img_shape", img_shape)
        kwargs["preset_name"] = kwargs.get("preset_name", "dit-s")

        return cls.with_preset(**kwargs)
