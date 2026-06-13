import pandas as pd
import torch
import torch.nn as nn
from self_attention import SelfAttention
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding

# =========================
# 1. DANE TRENINGOWE
# =========================

data = {
    "english": [
        "hello",
        "good morning",
        "i love cats",
        "thank you",
        "how are you"
    ],

    "polish": [
        "czesc",
        "dzien dobry",
        "kocham koty",
        "dziekuje",
        "jak sie masz"
    ]
}

df = pd.DataFrame(data)

# =========================
# 2. TWORZENIE SŁOWNIKA
# =========================

all_words = []

# Przechodzimy po wszystkich zdaniach angielskich
for sentence in df["english"]:

    # Dzielimy zdanie na słowa
    words = sentence.split()

    # Przechodzimy po każdym słowie
    for word in words:

        # Dodajemy słowo tylko jeśli jeszcze go nie ma
        if word not in all_words:
            all_words.append(word)

# =========================
# 3. PRZYPISYWANIE NUMERÓW
# =========================

word2idx = {
    "<PAD>": 0
}

for index, word in enumerate(all_words):

    word2idx[word] = index + 1      #+1 bo 0 jest zarezerwowane dla <PAD>

# =========================
# 4. SŁOWNIK I ZAKODOWANE ZDANIA
print("SŁOWNIK:\n")

for word, idx in word2idx.items():
    print(word, "->", idx)

print("\nZAKODOWANE ZDANIA:\n")

for sentence in df["english"]:

    encoded_sentence = []

    for word in sentence.split():

        encoded_sentence.append(word2idx[word])

    print(sentence, "->", encoded_sentence)

# =========================
# 5. PADDING

print("\nPADDING:\n")

encoded_sentences = []

for sentence in df["english"]:

    encoded = []

    for word in sentence.split():

        encoded.append(word2idx[word])

    encoded_sentences.append(encoded)

max_length = max(len(sentence) for sentence in encoded_sentences)

print("Najdluzsze zdanie ma dlugosc:", max_length)

padded_sentences = []

for sentence in encoded_sentences:

    padded = sentence.copy()

    while len(padded) < max_length:
        padded.append(0)

    padded_sentences.append(padded)

print("\nPo paddingu:\n")

for sentence in padded_sentences:
    print(sentence)

# =========================
#  6. EMBEDDING

# Zamieniamy listę na tensor PyTorch
input_tensor = torch.tensor(padded_sentences)

print("\nTensor wejściowy:")
print(input_tensor)

# Liczba słów w słowniku
vocab_size = len(word2idx)

# Rozmiar embeddingu
embedding_dim = 8

# Tworzymy warstwę embeddingu np 1 -> [... 8 liczb ...] dla każdego słowa w słowniku 
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Przekazujemy tensor wejściowy przez warstwę embeddingu i otrzymujemy tensor wyjściowy, gdzie każde słowo jest reprezentowane przez wektor o rozmiarze embedding_dim
#czyli zamiana identyfikatory słów na wektory liczb rzeczywistych
embedded = embedding(input_tensor)

print("\nPo embeddingu:")
print(embedded)

print("\nRozmiar:")
print(embedded.shape)

#wynik tu był tensor o wymiarach (liczba_zdań, max_length, embedding_dim) czyli (5, 3, 8) w naszym przypadku

# =========================
# POSITIONAL ENCODING
# =========================

positional_encoding = PositionalEncoding(
    embedding_dim=8
)

embedded = positional_encoding(
    embedded
)

print("\nPo positional encoding:")
print(embedded.shape)

# =========================
# SELF ATTENTION
# =========================

attention = MultiHeadAttention(
    embedding_dim=8
)

attention_output = attention(
    embedded
)

print("\nRozmiar attention_output:")
print(attention_output.shape)