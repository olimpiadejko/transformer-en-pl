import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)
        self.Wv = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        return attention_output