import torch
import torch.nn as nn

from transformer import Transformer

class CharNN(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, attention_dim, hidden_dim, head_num):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.transformer = Transformer(embedding_dim, attention_dim, hidden_dim, head_num)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        token_vector = self.token_embedding(x)

        positions = torch.arange(seq_len, device=x.device)
        positions_vector = self.position_embedding(positions)

        x = token_vector + positions_vector

        x = self.transformer(x)
        logits = self.output_layer(x)

        return logits


if __name__ == "__main__":
    model = CharNN(
        vocab_size=60,
        block_size=32,
        head_num=2
    )

    x = torch.randint(0, 60, (16, 32))

    logits = model(x)

    print("Вход:", x.shape)
    print("Выход:", logits.shape)
