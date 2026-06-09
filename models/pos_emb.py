import torch
import torch.nn as nn


class SinPosEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"

        half_dim = dim // 2
        exponent = torch.log(torch.tensor(max_period)) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim) * -exponent)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(-1) * self.inv_freq
        pos_emb = torch.cat((x.sin(), x.cos()), dim=-1)

        return pos_emb


class PatchSinPosEmbedding(SinPosEmbedding):
    def __init__(self, grid, dim, max_period=10000.0):
        super().__init__(dim // 2, max_period)
        assert len(grid) == 2, "Grid must be a tuple of length 2"

        grid_h, grid_w = grid
        ys, xs = torch.arange(grid_h), torch.arange(grid_w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        y_embed = super().forward(grid_y.flatten())
        x_embed = super().forward(grid_x.flatten())
        pos_emb = torch.cat((y_embed, x_embed), dim=-1).unsqueeze(
            0
        )  # [1, grid_h * grid_w, dim]
        self.register_buffer("pos_emb", pos_emb, persistent=False)

    def forward(self, x):
        return x + self.pos_emb
