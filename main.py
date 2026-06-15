import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from self_attention import SelfAttention
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from translator_model import TranslatorModel

# =========================
# 1. DANE TRENINGOWE
# =========================


data = {
    "english": [
        # Powitania i zwroty grzecznościowe
        "hello",
        "good morning",
        "thank you",
        "how are you",
        "good afternoon",
        "bye bye",
        
        # Zdania o kotach i psach (nauka rzeczowników i czasowników)
        "i love cats",
        "you love cats",
        "i love dogs",
        "you love dogs",
        "cats love dogs",
        "dogs love cats",
        
        # Proste opisy stanów
        "i am good",
        "you are good",
        "it is good",
        "i am here",
        "you are here"
    ],

    "polish": [
        # Powitania i zwroty grzecznościowe
        "czesc",
        "dzien dobry",
        "dziekuje",
        "jak sie masz",
        "dzien dobry",
        "pa pa",
        
        # Zdania o kotach i psach
        "kocham koty",
        "kochasz koty",
        "kocham psy",
        "kochasz psy",
        "koty kochaja psy",
        "psy kochaja koty",
        
        # Proste opisy stanów
        "jestem dobry",
        "jestes dobry",
        "to jest dobre",
        "jestem tutaj",
        "jestes tutaj"
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


# =========================
# 8. PĘTLA TRENINGOWA
# =========================


# ignorujemy <PAD> (indeks 0), aby model nie uczył się pustych miejsc
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 100

print("\nRozpoczęcie treningu:")
for epoch in range(EPOCHS):
    model.train() # Tryb treningowy
    
    # 1. Zerowanie gradientów
    optimizer.zero_grad()
    
    # 2. Forward pass (krok w przód)
    output = model(input_tensor)
    
    # 3. Obliczanie błędu (Loss)
    # output.view spłaszcza macierz do 2D: [ilość_słów, rozmiar_słownika]
    # target_tensor.view spłaszcza macierz docelową do 1D: [ilość_słów]
    loss = loss_fn(output.view(-1, output_vocab_size), target_tensor.view(-1))
    
    # 4. Backward pass (propagacja wsteczna)
    loss.backward()
    
    # 5. Aktualizacja wag
    optimizer.step()
    
    # Wypisywanie logów co 10 epok
    if (epoch + 1) % 10 == 0:
        print(f"Epoka: {epoch + 1:3d} | Błąd (Loss): {loss.item():.4f}")



# =========================
# 9. GENEROWANIE TŁUMACZENIA (INFERENCE)
# =========================

model.eval()  # Przełączamy model w tryb ewaluacji

# Odwracamy słownik polski, żeby móc szukać słowa po indeksie (id -> słowo)
idx2pol = {idx: word for word, idx in pol_word2idx.items()}

print("\n--- TEST TŁUMACZENIA ---")

with torch.no_grad():
    # Pobieramy końcowe przewidywania modelu dla naszych zdań wejściowych
    predictions = model(input_tensor)  # Kształt: [5, 3, 10]
    
    # Dla każdego słowa wybieramy indeks z najwyższą wartością logitu
    # argmax na wymiarze 2 (-1) sprawdza wartości w słowniku
    predicted_indices = torch.argmax(predictions, dim=-1)  # Kształt: [5, 3]

# Przetwarzamy każde zdanie na tekst
for i, sentence_indices in enumerate(predicted_indices):
    translated_words = []
    
    # Sprawdzamy, ile NIE-ZEROWYCH słów miało oryginalne zdanie angielskie
    original_length = (input_tensor[i] != 0).sum().item()
    
    # Pobieramy tylko tyle słów z predykcji, ile faktycznie było na wejściu
    valid_predictions = sentence_indices[:original_length].tolist()
    
    for idx in valid_predictions:
        if idx != 0:
            translated_words.append(idx2pol[idx])
            
    translated_sentence = " ".join(translated_words)
    original_sentence = df["english"].iloc[i]
    expected_sentence = df["polish"].iloc[i]
    
    print(f"\nAngielski: '{original_sentence}'")
    print(f"Oczekiwany: '{expected_sentence}'")
    print(f"Model przetłumaczył: '{translated_sentence}'")


# =========================
# 10. TESTOWANIE NOWYCH ZDAŃ
# =========================

print("\n--- AUTOMATYCZNY TEST NOWYCH ZDAŃ ---")

# Lista zupełnie nowych zdań (złożonych ze słów, które model już zna)
nowe_zdania_testowe = [
    "bye bye dogs",
    "dogs are here"
]

model.eval()

for zdanie in nowe_zdania_testowe:
    # 1. Tokenizacja i odfiltrowanie nieznanych słów
    oryginalne_slowa = [word for word in zdanie.split() if word in eng_word2idx]
    oryginalna_dlugosc = len(oryginalne_slowa)
    
    if oryginalna_dlugosc == 0:
        continue
        
    nowe_encoded = [eng_word2idx[word] for word in oryginalne_slowa]
    
    # 2. Nałożenie paddingu do stałej długości max_length_eng
    while len(nowe_encoded) < max_length_eng:
        nowe_encoded.append(0)
        
    # 3. Konwersja na tensor
    nowy_tensor = torch.tensor(nowe_encoded).unsqueeze(0)
    
    # 4. Przepuszczenie przez model
    with torch.no_grad():
        nowa_predykcja = model(nowy_tensor)
        nowe_indeksy = torch.argmax(nowa_predykcja, dim=-1).squeeze(0)
        
    # 5. Ucinanie predykcji do oryginalnej długości wejścia i dekodowanie
    valid_predictions = nowe_indeksy[:oryginalna_dlugosc].tolist()
    
    wynikowe_slowa = []
    for idx in valid_predictions:
        if idx != 0:
            wynikowe_slowa.append(idx2pol[idx])
            
    tlumaczenie = " ".join(wynikowe_slowa)
    
    print(f"Angielski: '{zdanie}'")
    print(f"Model przetłumaczył: '{tlumaczenie}'\n")


# =========================
# 11. INTERAKTYWNE TESTOWANIE Z TERMINALA
# =========================

print("\n=== INTERAKTYWNY TRANSLATOR ===")
print("Wpisz zdanie po angielsku i naciśnij Enter.")
print("Wpisz 'q' i naciśnij Enter, aby wyjść z programu.")

model.eval()

while True:
    # Pobieranie tekstu od użytkownika z terminala
    nowe_zdanie = input("\nAngielski > ").strip()
    
    # Warunek wyjścia z pętli
    if nowe_zdanie.lower() == 'q':
        print("Zamykanie translatora...")
        break
        
    if not nowe_zdanie:
        continue

    # 1. Tokenizacja i odfiltrowanie nieznanych słów
    oryginalne_slowa = [word for word in nowe_zdanie.split() if word in eng_word2idx]
    oryginalna_dlugosc = len(oryginalne_slowa)

    if oryginalna_dlugosc == 0:
        print("Model nie zna żadnego ze słów w tym zdaniu!")
        continue

    nowe_encoded = [eng_word2idx[word] for word in oryginalne_slowa]

    # 2. Nałożenie paddingu do stałej długości max_length_eng
    while len(nowe_encoded) < max_length_eng:
        nowe_encoded.append(0)

    # 3. Konwersja na tensor i dodanie wymiaru batcha
    nowy_tensor = torch.tensor(nowe_encoded).unsqueeze(0)

    # 4. Przepuszczenie przez model
    with torch.no_grad():
        nowa_predykcja = model(nowy_tensor) 
        nowe_indeksy = torch.argmax(nowa_predykcja, dim=-1).squeeze(0) 

    # 5. Ucinanie predykcji do oryginalnej długości wejścia i dekodowanie
    valid_predictions = nowe_indeksy[:oryginalna_dlugosc].tolist()

    wynikowe_slowa = []
    for idx in valid_predictions:
        if idx != 0: 
            wynikowe_slowa.append(idx2pol[idx])

    tlumaczenie = " ".join(wynikowe_slowa)

    print(f"Polski    > '{tlumaczenie}'")