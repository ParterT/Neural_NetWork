from sched import scheduler

import torch
import torch.nn as nn

from corpus_loader import load_data, get_batch
from config import *
from model import CharNN


@torch.no_grad()
def estimate_loss(model, data, fn_loss, device):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        batch_X, batch_Y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        logits = model(batch_X)
        loss = fn_loss(logits.reshape(-1, logits.size(-1)), batch_Y.reshape(-1))
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids, char_to_id, id_to_char = load_data()

    model = CharNN(
        vocab_size=len(char_to_id),
        block_size=BLOCK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        attention_dim=ATTENTION_DIM,
        hidden_dim=HIDDEN_DIM,
        head_num=HEAD_NUM,
        layers_num=LAYER_NUM,
        dropout=DROPOUT,
    ).to(device)

    fn_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=LEARNING_RATE*0.1)

    best_val_loss = float("inf")
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    for step in range(1, MAX_STEPS + 1):
        X_batch, Y_batch = get_batch(train_ids, BLOCK_SIZE, BATCH_SIZE)
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = fn_loss(logits.reshape(-1, logits.size(-1)), Y_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, train_ids, fn_loss, device)
            val_loss = estimate_loss(model, val_ids, fn_loss, device)

            print(f"Step: {step}/{MAX_STEPS}, \tTrain Loss: {train_loss:.4f}, \tVal Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "char_to_id": char_to_id,
                    "id_to_char": id_to_char,
                }, "models/best_model.pt")

# OLD BLOCK
"""
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
                "models/best_model(old).pt"
            )
        print(
            f"Epoch: {epoch + 1}/{EPOCHS}, \t"
            f"Train Loss: {train_loss:.4f}, \t"
            f"Val Loss: {val_loss.item():.4f}, \t"
            f"Val Accuracy: {val_accuracy.item():.2%}"
        )
"""

if __name__ == "__main__":
    train()
