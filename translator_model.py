import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock

class TranslatorModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, num_blocks=3):
        super().__init__()
        
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim) for _ in range(num_blocks)]
        )
        
        self.linear = nn.Linear(embedding_dim, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.linear(x)
        
        return x