import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from config import *
from corpus_loader import dataloader
from model import CharNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, Y_train, X_val, Y_val, char_to_id, id_to_char = dataloader(BLOCK_SIZE)


    train_dataset = TensorDataset(X_train, Y_train)

    train_data = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CharNN(
        vocab_size=len(char_to_id),
        block_size=BLOCK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        attention_dim=ATTENTION_DIM,
        hidden_dim=HIDDEN_DIM,
        head_num=HEAD_NUM
    ).to(device)

    fn_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_X, batch_Y in train_data:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = fn_loss(logits.reshape(-1, logits.size(-1)), batch_Y.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_data)

        model.eval()

        with torch.no_grad():
            X_val_device = X_val.to(device)
            Y_val_device = Y_val.to(device)

            val_logits = model(X_val_device)
            val_loss = fn_loss(val_logits.reshape(-1, val_logits.size(-1)), Y_val_device.reshape(-1))

            val_predictions = val_logits.argmax(dim=-1)
            val_accuracy = (val_predictions == Y_val_device).float().mean()

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                "model_state": model.state_dict(),
                "char_to_id": char_to_id,
                "id_to_char": id_to_char},
                "models/best_model.pt"
            )
        print(
            f"Epoch: {epoch + 1}/{EPOCHS}, \t"
            f"Train Loss: {train_loss:.4f}, \t"
            f"Val Loss: {val_loss.item():.4f}, \t"
            f"Val Accuracy: {val_accuracy.item():.2%}"
        )


if __name__ == "__main__":
    train()