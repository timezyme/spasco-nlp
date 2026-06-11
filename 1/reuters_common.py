"""Shared utilities for the Reuters newswire topic classification project.

Used by part-1 (baseline model and optimal-epoch selection), part-2
(improved architecture vs classical baseline), and part-3 (optimizer
comparison). Dataset: keras.datasets.reuters, 46 topics, multi-hot
encoding over the 10,000 most frequent words (Chollet ch. 3.5 protocol:
first 1,000 training samples held out as the validation set).
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
from keras import layers, models, optimizers
from keras.datasets import reuters
from keras.utils import to_categorical
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

NUM_WORDS = 10000
NUM_CLASSES = 46
VAL_SIZE = 1000
SEED = 42


@dataclass(frozen=True)
class ReutersData:
    """Train/validation/test splits: multi-hot inputs, one-hot and integer labels."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_int: np.ndarray
    y_val_int: np.ndarray
    y_test_int: np.ndarray


def vectorize_sequences(sequences, dimension: int = NUM_WORDS) -> np.ndarray:
    """Encode lists of word indices as multi-hot float32 vectors."""
    results = np.zeros((len(sequences), dimension), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


def load_reuters(num_words: int = NUM_WORDS, val_size: int = VAL_SIZE) -> ReutersData:
    """Load Reuters and split off the first `val_size` training samples as validation."""
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=num_words
    )

    x_all = vectorize_sequences(train_data, num_words)
    x_test = vectorize_sequences(test_data, num_words)
    y_all = to_categorical(train_labels, NUM_CLASSES)
    y_test = to_categorical(test_labels, NUM_CLASSES)

    return ReutersData(
        x_train=x_all[val_size:],
        x_val=x_all[:val_size],
        x_test=x_test,
        y_train=y_all[val_size:],
        y_val=y_all[:val_size],
        y_test=y_test,
        y_train_int=np.asarray(train_labels[val_size:]),
        y_val_int=np.asarray(train_labels[:val_size]),
        y_test_int=np.asarray(test_labels),
    )


def build_baseline_model(
    num_words: int = NUM_WORDS, num_classes: int = NUM_CLASSES
) -> keras.Model:
    """Chollet ch. 3.5 reference network: Dense 64-64 with rmsprop."""
    model = models.Sequential(
        [
            keras.Input(shape=(num_words,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_improved_model(
    num_words: int = NUM_WORDS,
    num_classes: int = NUM_CLASSES,
    optimizer=None,
) -> keras.Model:
    """Wider funnel (256-128-64) with batch norm and progressive dropout."""
    model = models.Sequential([keras.Input(shape=(num_words,))])
    for units, dropout in ((256, 0.4), (128, 0.3), (64, 0.2)):
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=optimizer if optimizer is not None else optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: keras.Model,
    data: ReutersData,
    epochs: int,
    batch_size: int,
    callbacks=None,
) -> keras.callbacks.History:
    """Fit on the training split, validating each epoch. Silent by convention."""
    return model.fit(
        data.x_train,
        data.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data.x_val, data.y_val),
        callbacks=callbacks,
        verbose=0,
    )


def best_epoch_from_history(history: keras.callbacks.History) -> tuple[int, float]:
    """Return (1-indexed epoch, value) of the maximum validation accuracy."""
    val_acc = history.history["val_accuracy"]
    best = int(np.argmax(val_acc))
    return best + 1, float(val_acc[best])


def predict_classes(model: keras.Model, x: np.ndarray) -> np.ndarray:
    return np.argmax(model.predict(x, verbose=0), axis=1)


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Accuracy, macro/weighted F1, and the full per-class report as text."""
    lines = [
        f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}",
        f"Macro F1:    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
        "",
        "Per-class report:",
        classification_report(y_true, y_pred, zero_division=0),
    ]
    return "\n".join(lines)


def majority_class_baseline(y_train_int: np.ndarray, y_test_int: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent training class."""
    majority = np.bincount(y_train_int).argmax()
    return float(np.mean(y_test_int == majority))


def shuffled_labels_baseline(y_test_int: np.ndarray, seed: int = SEED) -> float:
    """Chollet's random baseline: accuracy of shuffled test labels."""
    shuffled = np.random.default_rng(seed).permutation(y_test_int)
    return float(np.mean(y_test_int == shuffled))


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


def tfidf_logreg_baseline(
    num_words: int = NUM_WORDS, val_size: int = VAL_SIZE, seed: int = SEED
) -> dict:
    """Classical baseline: TF-IDF over word counts + logistic regression.

    Fits the TF-IDF transform on the training split only, then evaluates on
    the same validation/test splits the neural models use.
    """
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=num_words
    )
    counts_all = _count_matrix(train_data, num_words)
    counts_test = _count_matrix(test_data, num_words)

    counts_train, counts_val = counts_all[val_size:], counts_all[:val_size]
    y_train = np.asarray(train_labels[val_size:])
    y_val = np.asarray(train_labels[:val_size])
    y_test = np.asarray(test_labels)

    start = time.perf_counter()
    tfidf = TfidfTransformer()
    x_train = tfidf.fit_transform(counts_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    val_pred = clf.predict(tfidf.transform(counts_val))
    test_pred = clf.predict(tfidf.transform(counts_test))
    return {
        "val_accuracy": accuracy_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "fit_seconds": fit_seconds,
        "test_pred": test_pred,
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
        label=f"Best epoch ({best_epoch}, val acc {val_acc[best_epoch - 1]:.4f})",
    )
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, "bo-", label="Training loss", markersize=4)
    ax2.plot(epochs, val_loss, "r^-", label="Validation loss", markersize=4)
    ax2.axvline(x=best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str
) -> None:
    """Row-normalized confusion matrix; cells unannotated (46 classes)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    ax.set_title(title)
    ax.set_xlabel("Predicted topic")
    ax.set_ylabel("True topic")
    fig.tight_layout()
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
