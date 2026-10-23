"""Bi-LSTM model definition and GloVe embedding loader."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config


def build_vectorizer(train_texts) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=config.MAX_TOKENS,
        output_mode="int",
        output_sequence_length=config.MAX_LEN,
        standardize=None,  # text is already cleaned in src/preprocess.py
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(list(train_texts)).batch(1024))
    return vectorizer


def load_glove_matrix(vocab: list[str], glove_path: Path = config.GLOVE_PATH) -> np.ndarray | None:
    """Build an embedding matrix aligned to the vectorizer vocabulary."""
    if not glove_path.exists():
        print(f"[warn] GloVe file not found at {glove_path} - using random embeddings.")
        return None

    print(f"Loading GloVe vectors from {glove_path} ...")
    word_index = {w: i for i, w in enumerate(vocab)}
    matrix = np.random.normal(0, 0.1, (len(vocab), config.EMBED_DIM)).astype("float32")
    matrix[0] = 0.0  # padding token
    hits = 0
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, *vec = line.rstrip().split(" ")
            idx = word_index.get(word)
            if idx is not None and len(vec) == config.EMBED_DIM:
                matrix[idx] = np.asarray(vec, dtype="float32")
                hits += 1
    print(f"GloVe coverage: {hits:,}/{len(vocab):,} words ({hits / len(vocab):.1%})")
    return matrix


def build_bilstm(vocab_size: int, embedding_matrix: np.ndarray | None = None) -> tf.keras.Model:
    """Embedding -> Bi-LSTM -> GlobalMaxPool -> Dense -> Sigmoid  (input: int token ids)."""
    if embedding_matrix is not None:
        embedding = layers.Embedding(
            vocab_size,
            config.EMBED_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=config.TRAIN_EMBEDDINGS,
            mask_zero=False,
            name="glove_embedding",
        )
    else:
        embedding = layers.Embedding(vocab_size, config.EMBED_DIM, name="embedding")

    inputs = layers.Input(shape=(config.MAX_LEN,), dtype="int64", name="token_ids")
    x = embedding(inputs)
    x = layers.SpatialDropout1D(config.SPATIAL_DROPOUT)(x)
    x = layers.Bidirectional(layers.LSTM(config.LSTM_UNITS, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_fake")(x)

    model = tf.keras.Model(inputs, outputs, name="bilstm_fake_news")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_inference_model(vectorizer, trained_model) -> tf.keras.Model:
    """Wrap vectorizer + model so the saved file accepts raw (cleaned) strings."""
    text_in = layers.Input(shape=(1,), dtype="string", name="text")
    ids = vectorizer(text_in)
    out = trained_model(ids)
    return tf.keras.Model(text_in, out, name="fake_news_detector")
