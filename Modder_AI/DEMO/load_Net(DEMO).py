import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim=16, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


checkpoint = torch.load(
    "model.pth",
    map_location="cpu",
    weights_only=True
)

model = Net(
    vocab_size=len(checkpoint["char_to_id"]),
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_size"],
    hidden_size=checkpoint["hidden_size"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model Loaded.")
print("Saved Epoch:", checkpoint["epoch"])
print("Set Count:", len(checkpoint["char_to_id"]))