import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(self, x):

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        positions = torch.arange(
            seq_length,
            device=x.device
        )

        positions = positions.unsqueeze(0)

        positions = positions.repeat(
            batch_size,
            1
        )

        positions = positions.unsqueeze(-1)

        positions = positions.float()

        return x + positions