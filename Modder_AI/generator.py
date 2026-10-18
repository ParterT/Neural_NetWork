import torch

from config import *
from corpus_loader import encoder, decoder
from model import CharNN

def generator(path = "models/best_model.pt"):
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
        head_num=HEAD_NUM
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, char_to_id, id_to_char, device


@torch.no_grad()
def generate(model, char_to_id, id_to_char, device, prompt, length, temperature):
    ids = encoder(prompt, char_to_id)

    for _ in range(length):
        next_id = sample_text(model, ids, device, temperature)
        ids.append(next_id)

    return decoder(ids, id_to_char)

@torch.no_grad()
def sample_text(model, char_to_id, device, temperature):
    context = char_to_id[-BLOCK_SIZE:]
    x = torch.tensor([context], dtype=torch.long, device=device)

    logits = model(x)
    logits = logits[0, -1, :] / temperature
    probs = torch.softmax(logits, dim=-1)

    return torch.multinomial(probs, 1).item()

if __name__ == "__main__":
    model, char_to_id, id_to_char, device = generator()

    print(generate(model, char_to_id, id_to_char, device, prompt="Нейронная сеть ", length=LENGTH, temperature=TEMPERATURE))
