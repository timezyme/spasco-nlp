"""Shared utilities for the IMDB sentiment-classification parts of project 2.

Used by part-2 (bag-of-words MLP on a 200-word vocabulary, with measured
10,000-word and classical reference runs) and part-3 (Embedding + LSTM
sequence model). Dataset: keras.datasets.imdb, 25,000 balanced training
and 25,000 test reviews (Chollet ch. 3.5 protocol: first 10,000 training
samples held out as the validation set).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.datasets import imdb
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

VAL_SIZE = 10000
SEED = 42


@dataclass(frozen=True)
class ImdbSplits:
    """Validation-holdout splits plus the full training arrays.

    x_all/y_all cover all 25,000 training reviews; x_train/x_val are the
    last-15,000/first-10,000 partition of the same arrays (views, not
    copies). Final models retrain on x_all at the selected epoch count.
    """

    x_all: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_all: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_imdb_raw(num_words: int):
    """Raw index sequences and integer labels, vocabulary capped at num_words."""
    return imdb.load_data(num_words=num_words)


def vectorize_sequences(sequences, dimension: int) -> np.ndarray:
    """Encode lists of word indices as multi-hot float32 vectors."""
    results = np.zeros((len(sequences), dimension), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


def _to_splits(x_all: np.ndarray, x_test: np.ndarray, train_labels, test_labels) -> ImdbSplits:
    y_all = np.asarray(train_labels, dtype="float32")
    y_test = np.asarray(test_labels, dtype="float32")
    return ImdbSplits(
        x_all=x_all,
        x_train=x_all[VAL_SIZE:],
        x_val=x_all[:VAL_SIZE],
        x_test=x_test,
        y_all=y_all,
        y_train=y_all[VAL_SIZE:],
        y_val=y_all[:VAL_SIZE],
        y_test=y_test,
    )


def multi_hot_splits(num_words: int) -> ImdbSplits:
    """Multi-hot bag-of-words splits at the given vocabulary size (part 2)."""
    (train_data, train_labels), (test_data, test_labels) = load_imdb_raw(num_words)
    return _to_splits(
        vectorize_sequences(train_data, num_words),
        vectorize_sequences(test_data, num_words),
        train_labels,
        test_labels,
    )


def padded_splits(num_words: int, maxlen: int) -> ImdbSplits:
    """Pre-padded integer-sequence splits for the sequence model (part 3)."""
    (train_data, train_labels), (test_data, test_labels) = load_imdb_raw(num_words)
    return _to_splits(
        keras.utils.pad_sequences(train_data, maxlen=maxlen),
        keras.utils.pad_sequences(test_data, maxlen=maxlen),
        train_labels,
        test_labels,
    )


def build_dense_model(input_dim: int) -> keras.Model:
    """Chollet ch. 3.5 reference network: Dense 16-16-1 with rmsprop."""
    model = models.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def build_lstm_model(
    num_words: int, maxlen: int, embedding_dim: int, lstm_units: int
) -> keras.Model:
    """Embedding -> LSTM -> sigmoid. Embedding output is (batch, maxlen, embedding_dim)."""
    model = models.Sequential(
        [
            keras.Input(shape=(maxlen,)),
            layers.Embedding(num_words, embedding_dim),
            layers.LSTM(lstm_units),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_data=None,
    callbacks=None,
) -> keras.callbacks.History:
    """Fit wrapper. Silent by convention."""
    return model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=0,
    )


def best_epoch_from_history(history: keras.callbacks.History) -> tuple[int, float, float]:
    """(1-indexed epoch of minimum validation loss, val accuracy there, val loss there)."""
    val_loss = history.history["val_loss"]
    best = int(np.argmin(val_loss))
    return best + 1, float(history.history["val_accuracy"][best]), float(val_loss[best])


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent training class."""
    majority = np.bincount(y_train.astype(int)).argmax()
    return float(np.mean(y_test.astype(int) == majority))


def _count_matrix(sequences, dimension: int) -> sparse.csr_matrix:
    """Sparse word-count matrix from index sequences (keeps term frequencies)."""
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for seq in sequences:
        counts = Counter(seq)
        indices.extend(counts.keys())
        data.extend(counts.values())
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(len(sequences), dimension), dtype=np.float32
    )


def tfidf_logreg_baseline(num_words: int, seed: int = SEED) -> dict:
    """Classical baseline: TF-IDF over word counts + logistic regression.

    Same vocabulary cap and validation holdout as the neural models; the
    TF-IDF transform is fit on the training split only.
    """
    (train_data, train_labels), (test_data, test_labels) = load_imdb_raw(num_words)
    counts_all = _count_matrix(train_data, num_words)
    counts_test = _count_matrix(test_data, num_words)

    counts_train, counts_val = counts_all[VAL_SIZE:], counts_all[:VAL_SIZE]
    y_train = np.asarray(train_labels[VAL_SIZE:])
    y_val = np.asarray(train_labels[:VAL_SIZE])
    y_test = np.asarray(test_labels)

    start = time.perf_counter()
    tfidf = TfidfTransformer()
    x_train = tfidf.fit_transform(counts_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    return {
        "val_accuracy": accuracy_score(y_val, clf.predict(tfidf.transform(counts_val))),
        "test_accuracy": accuracy_score(y_test, clf.predict(tfidf.transform(counts_test))),
        "fit_seconds": fit_seconds,
    }


def plot_training_history(
    history: keras.callbacks.History,
    best_epoch: int,
    title: str,
    save_path: str,
) -> None:
    """Dual-panel accuracy/loss curves with the selected epoch marked."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc, "bo-", label="Training accuracy", markersize=4)
    ax1.plot(epochs, val_acc, "r^-", label="Validation accuracy", markersize=4)
    ax1.axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        label=f"Selected epoch ({best_epoch}, val acc {val_acc[best_epoch - 1]:.4f})",
    )
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, "bo-", label="Training loss", markersize=4)
    ax2.plot(epochs, val_loss, "r^-", label="Validation loss", markersize=4)
    ax2.axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        label=f"Selected epoch ({best_epoch}, val loss {val_loss[best_epoch - 1]:.4f})",
    )
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def versions_header() -> str:
    """Provenance block written at the top of every results file."""
    import sklearn
    import tensorflow as tf

    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: keras {keras.__version__}, tensorflow {tf.__version__}, "
            f"scikit-learn {sklearn.__version__}",
            f"Random seed: {SEED}",
        ]
    )
