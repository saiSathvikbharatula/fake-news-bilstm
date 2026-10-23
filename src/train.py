"""Train the Bi-LSTM fake news classifier and a TF-IDF + Logistic Regression baseline.

Usage (from the project root):
    python -m src.train                 # full training
    python -m src.train --sample 5000   # quick run on a subset
    python -m src.train --no-glove      # train embeddings from scratch
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

import config
from src.data_loader import load_dataset, split_dataset
from src.model import build_bilstm, build_inference_model, build_vectorizer, load_glove_matrix


def set_seeds(seed=config.SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def run_baseline(train_df, test_df) -> dict:
    print("\n=== Baseline: TF-IDF + Logistic Regression ===")
    tfidf = TfidfVectorizer(max_features=config.MAX_TOKENS, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(train_df["clean"])
    X_test = tfidf.transform(test_df["clean"])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_df["label"])
    probs = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(test_df["label"], probs)
    print(metrics)
    return metrics


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[key], marker="o", label=f"train {key}")
        ax.plot(history.history[f"val_{key}"], marker="o", label=f"val {key}")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Bi-LSTM Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_comparison(baseline, bilstm):
    names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, [baseline[n] for n in names], 0.4, label="TF-IDF + LogReg")
    plt.bar(x + 0.2, [bilstm[n] for n in names], 0.4, label="Bi-LSTM + GloVe")
    plt.xticks(x, [n.capitalize() for n in names])
    lo = min(min(baseline[n] for n in names), min(bilstm[n] for n in names))
    plt.ylim(max(0, lo - 0.05), 1.0)
    plt.title("Model Comparison (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="train on N random rows")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--no-glove", action="store_true", help="skip pre-trained GloVe")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    set_seeds()
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset(sample=args.sample)
    print(f"Class balance (1 = fake): {df['label'].value_counts(normalize=True).round(3).to_dict()}")
    train_df, val_df, test_df = split_dataset(df)

    baseline = None if args.skip_baseline else run_baseline(train_df, test_df)

    # ---- Vectorize ----
    print("\n=== Bi-LSTM ===")
    vectorizer = build_vectorizer(train_df["clean"])
    vocab = vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab):,}")

    def to_ids(texts, chunk=2048):
        # dtype=object avoids a fixed-width '<U...' array padded to the longest article
        texts = np.array(list(texts), dtype=object)
        return np.concatenate([vectorizer(tf.constant(texts[i:i + chunk])).numpy()
                               for i in range(0, len(texts), chunk)])

    X_train, X_val, X_test = to_ids(train_df["clean"]), to_ids(val_df["clean"]), to_ids(test_df["clean"])
    y_train, y_val, y_test = (train_df["label"].values, val_df["label"].values, test_df["label"].values)

    embedding_matrix = None if args.no_glove else load_glove_matrix(vocab)
    model = build_bilstm(len(vocab), embedding_matrix)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ]

    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=args.epochs, batch_size=config.BATCH_SIZE,
                        callbacks=callbacks, verbose=2)
    train_minutes = (time.time() - start) / 60

    # ---- Evaluate ----
    probs = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    bilstm = compute_metrics(y_test, probs)
    print("\nBi-LSTM test metrics:", bilstm)
    print(classification_report(y_test, preds, target_names=["Real", "Fake"], digits=4))

    plot_history(history)
    plot_confusion(y_test, preds)
    if baseline:
        plot_comparison(baseline, bilstm)

    # ---- Save ----
    inference_model = build_inference_model(vectorizer, model)
    inference_model.save(config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    results = {
        "dataset_rows": len(df),
        "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
        "vocab_size": len(vocab),
        "used_glove": embedding_matrix is not None,
        "epochs_trained": len(history.history["loss"]),
        "train_minutes": round(train_minutes, 2),
        "bilstm": bilstm,
        "baseline_tfidf_logreg": baseline,
    }
    if baseline:
        results["f1_improvement_pct_points"] = round((bilstm["f1"] - baseline["f1"]) * 100, 2)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {config.METRICS_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
