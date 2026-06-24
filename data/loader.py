"""Data loading and preprocessing for the workplace message checker."""

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/train.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data from the given path."""
    return pd.read_csv(path)


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Create a binary label: 1 if any toxicity flag is set, else 0."""
    toxic_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    return df[toxic_columns].max(axis=1)


def load_dataset(path: str = DATA_PATH):
    """
    Load, label, and split the dataset into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_raw_data(path)
    texts = df["comment_text"].astype(str)
    labels = build_labels(df)

    return train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
