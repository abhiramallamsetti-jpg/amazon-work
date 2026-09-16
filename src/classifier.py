"""
src/classifier.py

Intent classifier for the Day 4 AI support agent.

Reuses the exact Day 3 TF-IDF + Logistic Regression pipeline so that Day 4
behaviour matches the measured Day 3 baseline.  The model is trained ONLY on
``data/splits/train.csv`` -- the Golden Set is never used for fitting.

The classifier exposes:
    - predicted_intent
    - confidence (max probability)
    - top_intents (sorted probability list)
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import (
    CONFIG,
    SPLITS_DIR,
    PROCESSED_DIR,
    SUPPORT_CASES_CSV,
)

MODEL_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_classifier.pkl"


def _build_pipeline() -> Pipeline:
    """Recreate the exact Day 3 pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=CONFIG.classifier_max_features,
            stop_words=CONFIG.classifier_stop_words,
        )),
        ("classifier", LogisticRegression(
            random_state=CONFIG.classifier_random_state,
            max_iter=CONFIG.classifier_max_iter,
        )),
    ])


def train_classifier(train_df: pd.DataFrame | None = None) -> Pipeline:
    """
    Train the classifier on the training split.

    Parameters
    ----------
    train_df : DataFrame, optional
        Training data with ``customer_message`` and ``intent`` columns.
        If None, loads ``data/splits/train.csv``.

    Returns
    -------
    sklearn Pipeline
    """
    if train_df is None:
        train_df = pd.read_csv(SPLITS_DIR / "train.csv")

    X = train_df["customer_message"].fillna("").astype(str)
    y = train_df["intent"].astype(str)

    pipe = _build_pipeline()
    pipe.fit(X, y)
    return pipe


def save_classifier(pipe: Pipeline, path: Path | None = None) -> Path:
    """Persist a trained classifier to disk."""
    if path is None:
        path = MODEL_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return path


def load_classifier(path: Path | None = None) -> Pipeline:
    """Load a persisted classifier, or train one if no artifact exists."""
    if path is None:
        path = MODEL_ARTIFACT_PATH
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return train_classifier()


class IntentClassifier:
    """
    Wraps the TF-IDF + Logistic Regression pipeline with a clean predict API.
    """

    def __init__(self, pipeline: Pipeline | None = None):
        if pipeline is None:
            pipeline = load_classifier()
        self.pipeline = pipeline
        self._labels = list(pipeline.classes_)

    def predict(self, text: str) -> dict:
        """
        Classify a single customer message.

        Returns
        -------
        dict with keys:
            intent             -- predicted label
            confidence         -- max probability
            top_intents        -- list of {intent, probability} sorted desc
        """
        if text is None:
            text = ""
        text = str(text)

        X = [text]
        proba = self.pipeline.predict_proba(X)[0]
        labels = self._labels

        ranked = sorted(
            zip(labels, proba.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )
        top_intents = [
            {"intent": lbl, "probability": round(float(p), 4)}
            for lbl, p in ranked
        ]
        return {
            "intent": top_intents[0]["intent"],
            "confidence": round(float(top_intents[0]["probability"]), 4),
            "top_intents": top_intents,
        }


def main():
    """Train and persist the classifier, printing a short summary."""
    pipe = train_classifier()
    path = save_classifier(pipe)
    clf = IntentClassifier(pipe)
    sample = clf.predict("Where is my order? It is late.")
    print(f"Model saved to {path}")
    print(f"Labels ({len(pipe.classes_)}): {list(pipe.classes_)}")
    print(f"Sample prediction: {sample}")


if __name__ == "__main__":
    main()