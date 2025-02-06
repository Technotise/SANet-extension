import os
import json
import pandas as pd
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab=vocab
        self.word_to_idx={word: idx for idx, word in enumerate(vocab)}

    def encode(self, text):
        return [self.word_to_idx.get(word, 0) for word in text.split()]  # Convert words to indices

def build_vocab(label_csv, min_freq=2):
    labels = pd.read_csv(label_csv, delimiter="|")
    all_text = " ".join(labels["translation"])  # Spalte 'translation' enthält die Texte
    word_counts = Counter(all_text.split())
    vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    vocab += [word for word, count in word_counts.items() if count >= min_freq]
    return vocab

# Funktion zum Speichern des Vokabulars
def save_vocab(vocab, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vokabular gespeichert unter: {file_path}")

# Tokenizer und Vokabular erstellen und speichern
def prepare_tokenizers(train_label_csv, test_label_csv, finished_dir):
    # Train-Vokabular
    train_vocab = build_vocab(train_label_csv, min_freq=2)
    train_tokenizer = SimpleTokenizer(vocab=train_vocab)
    save_vocab(train_vocab, os.path.join(finished_dir, "train_vocab.json"))

    # Test-Vokabular
    test_vocab = build_vocab(test_label_csv, min_freq=2)
    test_tokenizer = SimpleTokenizer(vocab=test_vocab)
    save_vocab(test_vocab, os.path.join(finished_dir, "test_vocab.json"))

    return train_tokenizer, test_tokenizer