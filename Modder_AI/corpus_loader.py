from pathlib import Path
import torch


EXTENSIONS = (".txt", ".md", ".java")

# OLD BLOCK
"""
def load_text():
    direct = Path("Data_corpus")
    text = []

    for path in sorted(direct.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            text.append(path.read_text(encoding="utf-8", errors="ignore"))

    return "\n".join(text)
"""

def load_texts():
    direct = Path("Data_corpus")
    texts = []
    for path in sorted(direct.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))

    return texts

def load_text():
    return "\n".join(load_texts())

def load_data():
    texts = load_texts()
    char_to_id, id_to_char = char_mod("\n".join(texts))

    train_part, val_part = [], []

    for text in texts:
        cut = int(len(text) * 0.9)
        train_part.append(text[:cut])
        val_part.append(text[cut:])

    train_ids = torch.tensor(encoder("\n".join(train_part), char_to_id), dtype=torch.long)
    val_ids = torch.tensor(encoder("\n".join(val_part), char_to_id), dtype=torch.long)

    return train_ids, val_ids, char_to_id, id_to_char
def char_mod(text):
    chars = sorted(set(text))

    char_to_id = {char: i for i, char in enumerate(chars)}
    id_to_char = {i: char for i, char in enumerate(chars)}

    return char_to_id, id_to_char

def encoder(text, char_to_id):
    encoded_text = [char_to_id[char] for char in text]

    return encoded_text

def decoder(encoded_text, id_to_char):
    return "".join(id_to_char[id] for id in encoded_text)

def split_chunks(encoded_text, block_size, chunk_size=512, val_every=10):
    X_train, Y_train, X_val, Y_val = [], [], [], []

    for n, start in enumerate(range(0, len(encoded_text), chunk_size)):
        part = encoded_text[start:start + chunk_size]
        X, Y = seq(part, block_size)

        if n % val_every == val_every - 1:
            X_val += X
            Y_val += Y
        else:
            X_train += X
            Y_train += Y

    return X_train, Y_train, X_val, Y_val


def seq(encoded_text, block_size):
    X = []
    Y = []
    for i in range(len(encoded_text) - block_size):
        X.append(encoded_text[i: i + block_size])
        Y.append(encoded_text[i + 1: i+ block_size + 1])

    return X, Y

def get_batch(data, block_size, batch_size):
    starts = torch.randint(0, len(data) - block_size, (batch_size,))

    X = torch.stack([data[s:s + block_size] for s in starts.tolist()])
    Y = torch.stack([data[s + 1:s + block_size + 1] for s in starts.tolist()])

    return X, Y

def dataloader(block_size):
    text = load_text()

    char_to_id, id_to_char = char_mod(text)

    encoded_text = encoder(text, char_to_id)

    X_train, Y_train, X_val, Y_val = split_chunks(encoded_text, block_size)

    X_train = torch.tensor(X_train, dtype=torch.long)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.long)
    Y_val = torch.tensor(Y_val, dtype=torch.long)

    return X_train, Y_train, X_val, Y_val, char_to_id, id_to_char
