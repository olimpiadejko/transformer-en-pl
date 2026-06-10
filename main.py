import pandas as pd

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

import torch
import torch.nn as nn

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
# TWORZENIE QUERY, KEY I VALUE
# =========================

# Tworzymy 3 niezależne warstwy liniowe.
# Każda będzie uczyć się innych wag.

Wq = nn.Linear(8, 8)
Wk = nn.Linear(8, 8)
Wv = nn.Linear(8, 8)

# Z embeddingów tworzymy:
# Query (Q)
# Key (K)
# Value (V)

Q = Wq(embedded)
K = Wk(embedded)
V = Wv(embedded)

print("\nRozmiar Q:")
print(Q.shape)

print("\nRozmiar K:")
print(K.shape)

print("\nRozmiar V:")
print(V.shape)

# =========================
# ATTENTION SCORES
# =========================

# Porównujemy Query z Key.
# Dzięki temu model określa,
# które słowa są ze sobą powiązane.

scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)

print("\nRozmiar scores:")
print(scores.shape)

# =========================
# SOFTMAX
# =========================

attention_weights = torch.softmax(
    scores,
    dim=-1
)

print("\nRozmiar attention_weights:")
print(attention_weights.shape)

print("\nPrzykładowe attention weights dla pierwszego zdania:")
print(attention_weights[0])

# =========================
# ATTENTION OUTPUT
# =========================

attention_output = torch.matmul(
    attention_weights,
    V
)

print("\nRozmiar attention_output:")
print(attention_output.shape)