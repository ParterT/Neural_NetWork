import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, attention_dim)
        self.value = nn.Linear(embedding_dim, attention_dim)
        self.key = nn.Linear(embedding_dim, attention_dim)
        self.norm = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.T / math.sqrt(K.shape[-1])

        mask = torch.triu(
            torch.ones_like(scores, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        print("Веса внимания:\n", weights)

        result = weights @ V
        x += result
        x =  self.norm(x)

        x += self.ffn(x)
        x = self.norm2(x)
        return x


x = torch.tensor([[0.2, -0.4, 0.7, 0.1],
                 [0.5, 0.3, -0.2, 0.8],
                  [-0.1, 0.6, 0.4, -0.3]])

attention = TransformerBlock(embedding_dim=4, attention_dim=4)
output = attention(x)

print("Average Values:", output.mean(dim=-1))
print("Forms:", x.shape, ">>>", output.shape)

print("Inner Form:", x.shape)
print("Outer Form:", output.shape)
print("Result:")
print(output)