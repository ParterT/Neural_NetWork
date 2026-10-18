import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, embedding_dim, attention_dim, hidden_dim, head_num):
        super().__init__()

        assert attention_dim % head_num == 0

        self.num_head = head_num
        self.head_dim = attention_dim // head_num

        self.query = nn.Linear(embedding_dim, attention_dim)
        self.key = nn.Linear(embedding_dim, attention_dim)
        self.value = nn.Linear(embedding_dim, attention_dim)

        self.out = nn.Linear(attention_dim, embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        sec_len = x.shape[1]
        mask = torch.triu(
            torch.ones(sec_len, sec_len, device=x.device, dtype=torch.bool),
            diagonal=1)

        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        result = weights @ V

        result = result.transpose(1, 2).contiguous()
        result = result.view(batch_size, seq_len, self.num_head * self.head_dim)
        result = self.out(result)

        x = self.norm(x + result)
        x = self.norm2(x + self.ffn(x))

        return x


if __name__ == "__main__":
    block = Transformer(
        embedding_dim=32,
        attention_dim=32,
        hidden_dim=128,
        head_num=2
    )

    x = torch.randn(16, 32, 32)
    output = block(x)

    print("Вход:", x.shape)
    print("Выход:", output.shape)
