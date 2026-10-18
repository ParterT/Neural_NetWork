import torch
import torch.nn as nn

block_size = 4
embedding_dim = 4

position_embedding = nn.Embedding(block_size, embedding_dim)

positions = torch.arange(block_size)
position_vectors = position_embedding(positions)

print("Позиции:", positions)
print("Форма:", position_vectors.shape)
print(position_vectors)

token_embedding = nn.Embedding(10, embedding_dim)

token_ids = torch.tensor([3, 7, 2, 5])
token_vectors = token_embedding(token_ids)

x = token_vectors + position_vectors

print("Символьные векторы:", token_vectors.shape)
print("Позиционные векторы:", position_vectors.shape)
print("Итоговая форма:", x.shape)
print(x)