"""Predict whether a news article is fake.

Usage (from the project root):
    python -m src.predict "Paste the article text here"
    python -m src.predict --file article.txt
"""
import argparse

import numpy as np
import tensorflow as tf

import config
from src.preprocess import clean_text

_model = None


def load_model(path=config.MODEL_PATH):
    global _model
    if _model is None:
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `python -m src.train` first.")
        _model = tf.keras.models.load_model(path)
    return _model


def predict(text: str) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning - please provide a longer article.")
    p_fake = float(np.asarray(load_model()(tf.constant([[cleaned]])))[0][0])
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if label == "FAKE" else 1 - p_fake
    return {"label": label, "confidence": round(confidence, 4), "p_fake": round(p_fake, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="article text")
    parser.add_argument("--file", help="path to a .txt file containing the article")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide article text or --file")

    result = predict(text)
    print(f"Prediction: {result['label']}  (confidence {result['confidence']:.1%})")


if __name__ == "__main__":
    main()
