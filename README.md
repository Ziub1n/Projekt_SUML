# Workplace Message Checker

Aplikacja uczenia maszynowego, która analizuje czy wiadomość jest odpowiednia do wysłania w środowisku pracy.

## Co robi aplikacja

Wklej dowolną wiadomość do aplikacji — natychmiast sklasyfikuje ją jako **Odpowiednią** lub **Nieodpowiednią** dla środowiska pracy wraz z wynikiem pewności predykcji.

Model został wytrenowany na zbiorze danych [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) (~159 tys. oznaczonych wiadomości).

## Struktura projektu

```
workplace-message-checker/
├── data/
│   ├── loader.py        # wczytywanie i przetwarzanie danych
│   └── raw/
│       └── train.csv    # zbiór treningowy (Jigsaw)
├── model/
│   ├── train.py         # trenowanie modelu (TF-IDF + Regresja Logistyczna)
│   └── predict.py       # predykcja na nowych wiadomościach
├── app/
│   └── main.py          # aplikacja webowa Streamlit
├── models/              # zapisane pliki modelu (tworzone przy pierwszym uruchomieniu)
├── requirements.txt
└── README.md
```

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

## Ręczne trenowanie modelu (opcjonalnie)

```bash
python -m model.train
```

## Użyte technologie

| Warstwa | Technologia |
|---------|-------------|
| Dane    | pandas, scikit-learn |
| Model   | TF-IDF + Regresja Logistyczna (scikit-learn) |
| Aplikacja | Streamlit |
