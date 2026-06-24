"""Load a trained model and run predictions on new messages."""

import pickle
from sklearn.pipeline import Pipeline


MODEL_PATH = "models/classifier.pkl"


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """Load a trained pipeline from disk."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict_message(text: str, pipeline: Pipeline) -> dict:
    """
    Predict whether a message is appropriate for a workplace.

    Args:
        text: the message to evaluate
        pipeline: a trained sklearn Pipeline

    Returns:
        dict with keys: label (str), confidence (float), is_appropriate (bool)
    """
    probability = pipeline.predict_proba([text])[0]
    inappropriate_prob = float(probability[1])
    is_appropriate = inappropriate_prob < 0.5

    return {
        "is_appropriate": is_appropriate,
        "label": "Appropriate" if is_appropriate else "Inappropriate",
        "confidence": 1 - inappropriate_prob if is_appropriate else inappropriate_prob,
        "inappropriate_score": inappropriate_prob,
    }
