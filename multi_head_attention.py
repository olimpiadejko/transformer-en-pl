import torch
import torch.nn as nn

from self_attention import SelfAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.head1 = SelfAttention(embedding_dim)
        self.head2 = SelfAttention(embedding_dim)
        self.head3 = SelfAttention(embedding_dim)
        self.head4 = SelfAttention(embedding_dim)

        self.linear = nn.Linear(
            embedding_dim * 4,
            embedding_dim
        )

    def forward(self, x):

        h1 = self.head1(x)
        h2 = self.head2(x)
        h3 = self.head3(x)
        h4 = self.head4(x)

        combined = torch.cat(
            [h1, h2, h3, h4],
            dim=-1
        )

        output = self.linear(combined)

        return output