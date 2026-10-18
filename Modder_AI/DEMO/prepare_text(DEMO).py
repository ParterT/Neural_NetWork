import torch
from pathlib import Path
import torch.nn as nn

text = Path("../Data_corpus/corpus.txt").read_text(encoding="utf-8")
symbols = sorted(set(text))
char_to_id = {
    char: index
    for index, char in enumerate(symbols)
}
id_to_char = {
    index: char
    for char, index in char_to_id.items()
}
encoded_text = [char_to_id[char] for char in text]
block_size = 32
X = []
Y = []
for i in range(len(encoded_text) - block_size):
    context = encoded_text[i:i + block_size]
    target = encoded_text[i + block_size]
    X.append(context)
    Y.append(target)
X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)


class Net(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_size=16, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.fc1 = nn.Linear(block_size*embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

model = Net(vocab_size=len(symbols), block_size=block_size)

test_logits = model(X_tensor[:4])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    logits = model(X_tensor)
    loss = loss_fn(logits, Y_tensor)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
"""
    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )
"""

content = {
    "epoch": epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "char_to_id": char_to_id,
    "id_to_char": id_to_char,
    "block_size": block_size,
    "embedding_size": 16,
    "hidden_size": 128
}

torch.save(content, "model_v1.0.1.pth")
print("Model Saved")

with torch.no_grad():
    logits = model(X_tensor)

    prediction = logits.argmax(dim=1)

    accuracy = (prediction == Y_tensor).float().mean()

print(f"Accuracy: {accuracy.item():.2%}")


def generate_text(start_text, length=100):

    model.eval()

    context = [
        char_to_id[char]
        for char in start_text
        if char in char_to_id
    ]

    generated = start_text

    with torch.no_grad():
        for _ in range(length):
            current_context = context[-block_size:]
            x = torch.tensor([current_context], dtype=torch.long)

            logits = model(x)

            temperature = 1.0
            probabilities = torch.softmax(logits[0] / temperature, dim=0)
            next_id = torch.multinomial(probabilities, 1).item()

            next_char = id_to_char[next_id]

            generated += next_char
            context.append(next_id)

    return generated

"""
model.eval()

start_text = "Космический огурец программирует межгалактический чайник"

context = [
    char_to_id[char]
    for char in start_text
    if char in char_to_id
]

x = torch.tensor([context[-block_size:]], dtype=torch.long)

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)

    top_prob, top_ids = torch.topk(probabilities[0], 3)
    for id_char, probability in zip(top_ids.tolist(), top_prob.tolist()):
        char = id_to_char[id_char]
        print(f"Symbol: {char!r}, probability: {probability:.2%}")

print("Logits:", logits)
print("Probability:", probabilities)
print("Sum:", probabilities.sum().item())
"""

print(generate_text("Космический огурец программирует межгалактический чайник", length=100))
