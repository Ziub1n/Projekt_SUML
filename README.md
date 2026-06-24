# Workplace Message Checker

Aplikacja uczenia maszynowego, która analizuje czy wiadomość jest odpowiednia do wysłania w środowisku pracy.

## Opis aplikacji

Aplikacja klasyfikuje wiadomości tekstowe jako **Odpowiednie** lub **Nieodpowiednie** dla środowiska pracy. Użytkownik wpisuje dowolną wiadomość, a system zwraca predykcję wraz z wynikiem pewności (%).

Model został wytrenowany na zbiorze danych [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) (~159 tys. oznaczonych wiadomości).

## Struktura projektu

```
workplace-message-checker/
├── data/
│   ├── loader.py        # wczytywanie i przetwarzanie danych
│   └── raw/
│       └── train.csv    # zbiór treningowy (Jigsaw) — patrz sekcja "Dane"
├── model/
│   ├── train.py         # trenowanie modelu (TF-IDF + Regresja Logistyczna)
│   └── predict.py       # predykcja na nowych wiadomościach
├── app/
│   └── main.py          # aplikacja webowa Streamlit
├── models/              # zapisane pliki modelu (tworzone przy pierwszym uruchomieniu)
├── requirements.txt
└── README.md
```

## Dane treningowe

Zbiór danych pochodzi z konkursu Kaggle: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

Kroki:
1. Pobierz plik `train.csv.zip` ze strony konkursu.
2. Wypakuj go i umieść plik `train.csv` w katalogu `data/raw/train.csv`.

## Wymagania

- Python 3.10+
- Plik z danymi treningowymi umieszczony w `data/raw/train.csv`

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie aplikacji

Uruchom z **głównego katalogu projektu**:

```bash
streamlit run app/main.py
```

Przy pierwszym uruchomieniu model zostanie automatycznie wytrenowany (ok. 1 minuta). Kolejne uruchomienia wczytują zapisany model natychmiastowo.

## Sposób działania

1. Uruchom aplikację komendą powyżej.
2. W polu tekstowym wpisz lub wklej wiadomość, którą chcesz sprawdzić.
3. Kliknij przycisk **Check message**.
4. Aplikacja zwróci wynik:
   - **Appropriate** — wiadomość jest odpowiednia do wysłania w pracy
   - **Inappropriate** — wiadomość może zawierać nieodpowiednie treści
   - Wynik pewności predykcji w procentach (%)
   - Pasek z ogólnym wskaźnikiem nieodpowiedniości

## Model ML

| Parametr | Wartość |
|----------|---------|
| Algorytm | Regresja Logistyczna |
| Ekstrakcja cech | TF-IDF (50 000 cech, unigramy + bigramy) |
| Zbiór treningowy | 127 656 próbek |
| Zbiór testowy | 31 915 próbek |
| Dokładność (accuracy) | 94% |
| F1-score (nieodpowiednie) | 75% |
| Wynik pylint | 9.76 / 10 |

## Ręczne trenowanie modelu (opcjonalnie)

```bash
python -m model.train
```

## Użyte technologie

| Warstwa | Technologia |
|---------|-------------|
| Dane | pandas, scikit-learn |
| Model | TF-IDF + Regresja Logistyczna (scikit-learn) |
| Aplikacja | Streamlit |

## Autorzy

- Ziub1n (s28486)
