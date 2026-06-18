import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from self_attention import SelfAttention
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from translator_model import TranslatorModel

def main():
    # wczytanie datasetu
    df = pd.read_csv("dataset2.csv")
    eng_word2idx, pol_word2idx = build_vocab(df)

    # Podział na zbiór treningowy (80%) i testowy (20%)
    # random_state=42 zapewnia, że za każdym razem podział będzie taki sam
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    max_len_eng = max(len(s.split()) for s in df["english"])
    print("Najdluzsze zdanie EN ma dlugosc:", max_len_eng)

    max_len_pol = max(len(s.split()) for s in df["polish"])
    print("Najdluzsze zdanie PL ma dlugosc:", max_len_pol)

    # Znajdujemy największą długość ze wszystkich zdań w obu językach
    global_max_len = max(max_len_eng, max_len_pol)

    train_input_tensor = create_tensor(train_df["english"], eng_word2idx, global_max_len)
    train_target_tensor = create_tensor(train_df["polish"], pol_word2idx, global_max_len)

    test_input_tensor = create_tensor(test_df["english"], eng_word2idx, global_max_len)
    test_target_tensor = create_tensor(test_df["polish"], pol_word2idx, global_max_len)

    embedding_dim = 8

    input_vocab_size = len(eng_word2idx)
    output_vocab_size = len(pol_word2idx)

    model = TranslatorModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        num_blocks=3
    )

    #trening
    EPOCHS = 300
    losses = train_loop(model, EPOCHS, train_input_tensor, train_target_tensor, output_vocab_size)

    #wykres loss
    create_loss_figure(losses)

    #accuracy
    accuracy = evaluate(model, test_input_tensor, test_target_tensor)

    #inferencja
    #print("\n--- TŁUMACZENIE ZDAŃ TRENINGOWYCH---")
    #inference(model, train_df, pol_word2idx, train_input_tensor)

    print("\n--- TŁUMACZENIE ZDAŃ TESTOWYCH---")
    inference(model, test_df, pol_word2idx, test_input_tensor)

    

def build_vocab(df):
    #tworzenie słownikow EN i PL na podstawie calego datasetu
    eng_words = []
    pol_words = []

    for sentence in df["english"]:
        for word in sentence.split():
            if word not in eng_words:
                eng_words.append(word)

    for sentence in df["polish"]:
        for word in sentence.split():
            if word not in pol_words:
                pol_words.append(word)
    
    #indeksowanie
    eng_word2idx = {"<PAD>": 0}
    pol_word2idx = {"<PAD>": 0}

    for index, word in enumerate(eng_words):
        eng_word2idx[word] = index + 1      #+1 bo 0 jest zarezerwowane dla <PAD>

    for index, word in enumerate(pol_words):
        pol_word2idx[word] = index + 1

    return eng_word2idx, pol_word2idx


def create_tensor(sentences, word2idx, max_length):
    #kodowanie i padding
    encoded_sentences = []
    for sentence in sentences:
        encoded = []
        for word in sentence.split():
            encoded.append(word2idx[word])
        encoded_sentences.append(encoded)
 
    padded_sentences = []
    for sentence in encoded_sentences:
        padded = sentence.copy()
        while len(padded) < max_length:
            padded.append(0)
        padded_sentences.append(padded)

    tensor = torch.tensor(padded_sentences)
    return tensor


def train_loop(model, epoch_num, input_tensor, target_tensor, vocab_size):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    losses = []
    print("\nRozpoczęcie treningu:")
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output.view(-1, vocab_size), target_tensor.view(-1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoka: {epoch + 1:3d} | Błąd (Loss): {loss.item():.4f}")

    return losses


def create_loss_figure(losses):
    plt.figure(figsize=(8,5))
    plt.plot(losses)

    plt.title("Przebieg uczenia modelu")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.show()


def evaluate(model, input_tensor, target_tensor):
    model.eval()

    with torch.no_grad():
        predictions = model(input_tensor)

        predicted_indices = torch.argmax(
            predictions,
            dim=-1
        )

        correct = 0
        total = 0

        for i in range(target_tensor.shape[0]):
            for j in range(target_tensor.shape[1]):

                target = target_tensor[i, j].item()

                if target != 0:
                    total += 1
                    prediction = predicted_indices[i, j].item()

                    if prediction == target:
                        correct += 1

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {accuracy:.2f}%")
    return accuracy


def inference(model, df, pol_word2idx, input_tensor):
    model.eval()

    idx2pol = {idx: word for word, idx in pol_word2idx.items()}

    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_indices = torch.argmax(predictions, dim=-1)

    for i, sentence_indices in enumerate(predicted_indices):
        translated_words = []
        original_length = (input_tensor[i] != 0).sum().item()
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


if __name__ == "__main__":
    main()