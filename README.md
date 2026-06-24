# Workplace Message Checker

A machine learning application that analyses whether a message is appropriate to send in a professional work environment.

## What it does

Paste any message into the app — it will instantly classify it as **Appropriate** or **Inappropriate** for the workplace, along with a confidence score.

The model was trained on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset (~159k labelled messages).

## Project structure

```
workplace-message-checker/
├── data/
│   ├── loader.py        # data loading and preprocessing
│   └── raw/
│       └── train.csv    # training dataset (Jigsaw)
├── model/
│   ├── train.py         # model training (TF-IDF + Logistic Regression)
│   └── predict.py       # inference on new messages
├── app/
│   └── main.py          # Streamlit web application
├── models/              # saved model files (created on first run)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- The training data file must be placed at `data/raw/train.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Running the app

Run from the **project root** directory:

```bash
streamlit run app/main.py
```

The model will be trained automatically on the first launch (takes ~1 minute). Subsequent launches load the saved model instantly.

## Training the model manually (optional)

```bash
python -m model.train
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Data  | pandas, scikit-learn |
| Model | TF-IDF + Logistic Regression (scikit-learn) |
| App   | Streamlit |
