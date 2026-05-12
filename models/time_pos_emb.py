import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        half_dim = dim // 2
        embeddings_buffer = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings_buffer = torch.exp(torch.arange(half_dim) * -embeddings_buffer)
        self.register_buffer('embeddings_buffer', embeddings_buffer)

    def forward(self, time):
        embeddings = time[:, None] * self.embeddings_buffer[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings
