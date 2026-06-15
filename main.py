import pandas as pd
import torch
import torch.nn as nn
from self_attention import SelfAttention
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from translator_model import TranslatorModel

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

eng_words = []
pol_words = []

# Przechodzimy po wszystkich zdaniach angielskich
for sentence in df["english"]:

    # Dzielimy zdanie na słowa
    words = sentence.split()

    # Przechodzimy po każdym słowie
    for word in words:

        # Dodajemy słowo tylko jeśli jeszcze go nie ma
        if word not in eng_words:
            eng_words.append(word)

#to samo PL
for sentence in df["polish"]:
    for word in sentence.split():
        if word not in pol_words:
            pol_words.append(word)

# =========================
# 3. PRZYPISYWANIE NUMERÓW
# =========================

eng_word2idx = {
    "<PAD>": 0
}

for index, word in enumerate(eng_words):
    eng_word2idx[word] = index + 1      #+1 bo 0 jest zarezerwowane dla <PAD>


pol_word2idx = {"<PAD>": 0}

for index, word in enumerate(pol_words):
    pol_word2idx[word] = index + 1

input_vocab_size = len(eng_word2idx)
output_vocab_size = len(pol_word2idx)

# =========================
# 4. SŁOWNIK I ZAKODOWANE ZDANIA
# =========================

print("SŁOWNIK ANGIELSKI:\n")
for word, idx in eng_word2idx.items():
    print(word, "->", idx)

print("\nSŁOWNIK POLSKI:\n")
for word, idx in pol_word2idx.items():
    print(word, "->", idx)

print("\nZAKODOWANE ZDANIA ANGIELSKIE:\n")
for sentence in df["english"]:
    encoded_sentence = []
    for word in sentence.split():
        encoded_sentence.append(eng_word2idx[word])
    print(sentence, "->", encoded_sentence)

print("\nZAKODOWANE ZDANIA POLSKIE:\n")
for sentence in df["polish"]:
    encoded_sentence = []
    for word in sentence.split():
        encoded_sentence.append(pol_word2idx[word])
    print(sentence, "->", encoded_sentence)

# =========================
# 5. PADDING
# =========================

print("\nPADDING:\n")

# --- KODOWANIE I PADDING EN ---
eng_encoded_sentences = []
for sentence in df["english"]:
    encoded = []
    for word in sentence.split():
        encoded.append(eng_word2idx[word])
    eng_encoded_sentences.append(encoded)

max_length_eng = max(len(sentence) for sentence in eng_encoded_sentences)
print("Najdluzsze zdanie EN ma dlugosc:", max_length_eng)

eng_padded_sentences = []
for sentence in eng_encoded_sentences:
    padded = sentence.copy()
    while len(padded) < max_length_eng:
        padded.append(0)
    eng_padded_sentences.append(padded)

# --- KODOWANIE I PADDING PL ---
pol_encoded_sentences = []
for sentence in df["polish"]:
    encoded = []
    for word in sentence.split():
        encoded.append(pol_word2idx[word])
    pol_encoded_sentences.append(encoded)

max_length_pol = max(len(sentence) for sentence in pol_encoded_sentences)
print("Najdluzsze zdanie PL ma dlugosc:", max_length_pol)

pol_padded_sentences = []
for sentence in pol_encoded_sentences:
    padded = sentence.copy()
    while len(padded) < max_length_pol:
        padded.append(0)
    pol_padded_sentences.append(padded)

print("\nPo paddingu EN:\n")
for sentence in eng_padded_sentences:
    print(sentence)

print("\nPo paddingu PL:\n")
for sentence in pol_padded_sentences:
    print(sentence)

# =========================
#  6. TENSORY
# =========================

# Zamieniamy listę na tensor PyTorch
input_tensor = torch.tensor(eng_padded_sentences)
target_tensor = torch.tensor(pol_padded_sentences)

print("\nTensor wejściowy (EN):")
print(input_tensor)
print("\nTensor docelowy (PL):")
print(target_tensor)



# =========================
# 7. INICJALIZACJA MODELU
# =========================

embedding_dim = 8

model = TranslatorModel(
    input_vocab_size=input_vocab_size,
    output_vocab_size=output_vocab_size,
    embedding_dim=embedding_dim,
    num_blocks=3
)

# Testowy przepływ danych
model.eval()
with torch.no_grad():
    output = model(input_tensor)

print("\nRozmiar wyjścia z modelu:")
print(output.shape) # Spodziewany wynik: [5, 3, 10]