import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.attention = MultiHeadAttention(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.ff = FeedForward(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)
        
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x