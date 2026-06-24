"""Train and save the workplace message classification model."""

import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from data.loader import load_dataset


MODEL_PATH = "models/classifier.pkl"


def build_pipeline() -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def train_model(model_path: str = MODEL_PATH) -> Pipeline:
    """
    Train the model on the dataset and save it to disk.

    Args:
        model_path: where to save the trained pipeline (.pkl)

    Returns:
        Trained pipeline
    """
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_dataset()

    print(f"Training on {len(x_train)} samples...")
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    print("\nEvaluation on test set:")
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=["Appropriate", "Inappropriate"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)

    print(f"Model saved to {model_path}")
    return pipeline


if __name__ == "__main__":
    train_model()
