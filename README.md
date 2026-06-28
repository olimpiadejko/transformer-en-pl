# Uproszczony T³umacz Angielsko-Polski (Transformer Encoder-Only)

Projekt implementuje uproszczon¹ architekturê sieci neuronowej typu **Transformer (Encoder-Only)** od zera w bibliotece PyTorch. Model realizuje zadanie t³umaczenia maszynowego z jêzyka angielskiego na jêzyk polski w mapowaniu sekwencji typu 1:1.

---

## Technologie

* **Jêzyk:** Python 
* **Biblioteka g³ówna:** PyTorch (implementacja sieci i treningu od zera)
* **Biblioteki wspomagaj¹ce:** Pandas (obs³uga danych), Scikit-learn (podzia³ danych), Matplotlib (wykres b³êdu)

---

## Architektura Modelu 

1. **Tokenizacja:** Podzia³ tekstu po spacjach i zamiana na numeryczne indeksy s³ownika.
2. **Warstwa Embedding:** Mapowanie indeksów na gêste, 8-wymiarowe wektory znaczeniowe.
3. **Positional Encoding:** Uproszczone dodanie numeru pozycji do wektora s³owa (informacja o szyku zdania).
4. **Multi-Head Attention (4 g³owice Self-Attention):** Równoleg³e obliczanie macierzy *Query*, *Key*, *Value* w celu analizy kontekstu i powi¹zañ miêdzy s³owami.
5. **FeedForward:** Nieliniowe przetwarzanie wektorów przez dwie warstwy liniowe z aktywacj¹ ReLU.
6. **Warstwa Wyjœciowa (Klasyfikator):** Liniowe rzutowanie wyników na pe³ny rozmiar polskiego s³ownika w celu predykcji s³ów.

---

## Instrukcja Uruchomienia

Aby uruchomiæ projekt lokalnie, wykonaj poni¿sze kroki w swoim terminalu:

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/olimpiadejko/transformer-en-pl.git
cd transformer-en-pl

# 2. Utwórz wirtualne œrodowisko (zalecane)
python -m venv venv

# 3. Aktywuj œrodowisko
# Na systemach Windows:
venv\Scripts\activate
# Na systemach macOS/Linux:
source venv/bin/activate

# 4. Zainstaluj wymagane biblioteki
pip install torch pandas scikit-learn matplotlib

# 5. Uruchom g³ówny program ucz¹cy i testuj¹cy model
python main.py
```

---

## Wyniki i Ewaluacja

Jakoœæ modelu zosta³a zweryfikowana na wydzielonym zbiorze testowym przy u¿yciu metryki Accuracy.

* **Przebieg uczenia:** Model by³ trenowany przez 100 epok przy u¿yciu optymalizatora Adam (learning rate: 0.005) oraz funkcji straty CrossEntropyLoss z pomijaniem tokenów paddingu. Wartoœæ b³êdu (Loss) spad³a z oko³o 3.7 na pocz¹tku treningu do 0.3349 po 100 epokach.
* **Dok³adnoœæ (Accuracy):** Model osi¹gn¹³ dok³adnoœæ 71,74%. Oznacza to, ¿e oko³o 72% tokenów w zbiorze testowym zosta³o poprawnie przewidzianych z pominiêciem paddingu.
* **Przyk³ady dzia³añ:** 
  * cats love you -> koty kochaja cie (T³umaczenie poprawne)
  * cats hate cats -> koty nienawidza nienawidza (B³¹d dopasowania zaimka)
