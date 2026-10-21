import torch

from config import *
from corpus_loader import encoder, decoder
from model import CharNN

def generator(path = "models/best_model_dropout.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    char_to_id = checkpoint["char_to_id"]
    id_to_char = checkpoint["id_to_char"]

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

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, char_to_id, id_to_char, device


@torch.no_grad()
def generate(model, char_to_id, id_to_char, device, prompt, length, temperature, top_k=None):
    ids = encoder(prompt, char_to_id)

    for _ in range(length):
        next_id = sample_text(model, ids, device, temperature)
        ids.append(next_id)

    return decoder(ids, id_to_char)

@torch.no_grad()
def sample_text(model, char_to_id, device, temperature, top_k=None):
    context = char_to_id[-BLOCK_SIZE:]
    x = torch.tensor([context], dtype=torch.long, device=device)

    logits = model(x)
    logits = logits[0, -1, :] / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

if __name__ == "__main__":
    model, char_to_id, id_to_char, device = generator()

    for prompt in ["public class ", "# Registering "]:
        print(generate(model, char_to_id, id_to_char, device, prompt, LENGTH, TEMPERATURE, 10))
        print("-"*40)
