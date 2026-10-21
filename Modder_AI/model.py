import torch
import torch.nn as nn

from transformer import Transformer

class CharNN(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, attention_dim, hidden_dim, head_num, layers_num, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        """
        self.transformer = Transformer(embedding_dim, attention_dim, hidden_dim, head_num)
        """

        self.blocks = nn.Sequential(
            *[Transformer(embedding_dim, attention_dim, hidden_dim, head_num, dropout) for _ in range(layers_num)]
        )
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        token_vector = self.token_embedding(x)

        positions = torch.arange(seq_len, device=x.device)
        positions_vector = self.position_embedding(positions)

        x = token_vector + positions_vector

        x = self.blocks(x)
        logits = self.output_layer(x)

        return logits
