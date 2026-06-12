"""Part 3: MNIST CNN, the Chollet ch. 5.1 baseline vs a regularized variant.

Trains two architectures under one protocol (seeded, early stopping on
validation loss, single test evaluation each): the book's exact
Conv-Pool stack, and the same stack with batch normalization and
dropout. Reports the comparison, plots the winner's training history
and confusion matrix, and walks the baseline's layer output shapes to
answer (c): why Conv2D(32, (3, 3)) turns 28x28x1 into 26x26x32.
Writes part-3_output.txt and both PNGs next to this script; saves no
model files.
"""

import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
MAX_EPOCHS = 12
BATCH_SIZE = 128
SEED = 42


def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_train = train_images.reshape((-1, 28, 28, 1)).astype("float32") / 255
    x_test = test_images.reshape((-1, 28, 28, 1)).astype("float32") / 255
    return (
        x_train,
        to_categorical(train_labels),
        x_test,
        to_categorical(test_labels),
        test_labels,
    )


def build_baseline_cnn() -> keras.Model:
    """Chollet ch. 5.1 architecture, unchanged."""
    model = models.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_regularized_cnn() -> keras.Model:
    """Same conv skeleton with batch normalization and dropout added."""
    model = models.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_and_evaluate(build_fn, x_train, y_train, x_test, y_test) -> dict:
    """Seeded fit with early stopping; one test evaluation on best weights."""
    keras.utils.set_random_seed(SEED)
    model = build_fn()
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ],
        verbose=0,
    )
    fit_seconds = time.perf_counter() - start
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "val_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "params": model.count_params(),
        "fit_seconds": fit_seconds,
    }


def comparison_table(rows: dict[str, dict]) -> str:
    header = (
        f"{'Model':<26} {'Params':>9} {'Best ep':>8} {'Val acc':>8} "
        f"{'Test acc':>9} {'Fit (s)':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        lines.append(
            f"{name:<26} {r['params']:>9,} {r['best_epoch']:>8} "
            f"{r['val_accuracy']:>8.4f} {r['test_accuracy']:>9.4f} "
            f"{r['fit_seconds']:>8.1f}"
        )
    return "\n".join(lines)


def shape_walk(model: keras.Model) -> str:
    """(c): the baseline's actual layer output shapes plus the formula."""
    lines = [
        "Layer output shapes (baseline model, batch dimension omitted):",
    ]
    for layer in model.layers:
        shape = tuple(layer.output.shape[1:])
        lines.append(f"  {layer.__class__.__name__:<18} -> {shape}")
    lines += [
        "",
        "The first Conv2D maps 28x28x1 to 26x26x32: with 'valid' padding and",
        "stride 1, each spatial dimension shrinks to (28 - 3)/1 + 1 = 26 because",
        "a 3x3 window only fits 26 positions along a 28-pixel axis, and the 32",
        "filters each produce one 26x26 feature map.",
    ]
    return "\n".join(lines)


def plot_history(result: dict, title: str, save_path: Path) -> None:
    history = result["history"].history
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in ((ax1, "accuracy", "Accuracy"), (ax2, "loss", "Loss")):
        ax.plot(epochs, history[metric], "bo-", label=f"Training {label.lower()}", markersize=4)
        ax.plot(
            epochs,
            history[f"val_{metric}"],
            "r^-",
            label=f"Validation {label.lower()}",
            markersize=4,
        )
        ax.axvline(
            x=result["best_epoch"],
            color="green",
            linestyle="--",
            label=f"Selected epoch ({result['best_epoch']})",
        )
        ax.set_title(f"Training and Validation {label}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(model: keras.Model, x_test, test_labels_int, save_path: Path) -> int:
    """Annotated 10x10 confusion matrix; returns the misclassified count."""
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(test_labels_int, predictions)
    misclassified = int(cm.sum() - np.trace(cm))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Images")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("MNIST test confusion matrix (winning model)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return misclassified


def versions_header() -> str:
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


def main() -> None:
    x_train, y_train, x_test, y_test, test_labels_int = prepare_data()

    results = {
        "Baseline CNN (ch. 5.1)": train_and_evaluate(
            build_baseline_cnn, x_train, y_train, x_test, y_test
        ),
        "Regularized CNN (BN+drop)": train_and_evaluate(
            build_regularized_cnn, x_train, y_train, x_test, y_test
        ),
    }
    winner_name = max(results, key=lambda k: results[k]["val_accuracy"])
    winner = results[winner_name]

    plot_history(
        winner,
        f"MNIST {winner_name} - Training History",
        HERE / "training_history.png",
    )
    misclassified = plot_confusion(
        winner["model"], x_test, test_labels_int, HERE / "confusion_matrix.png"
    )

    report = "\n".join(
        [
            "PART 3 - MNIST CNN: BASELINE VS REGULARIZED VARIANT",
            "=" * 60,
            versions_header(),
            "",
            f"Protocol: 54k/6k train/val split (validation_split 0.1), max "
            f"{MAX_EPOCHS} epochs,",
            f"batch {BATCH_SIZE}, early stopping on val loss (patience 3, best "
            "weights restored),",
            "rmsprop. Winner picked on validation accuracy; each model touches "
            "the test set once.",
            "",
            comparison_table(results),
            "",
            f"Winner on validation: {winner_name}",
            f"(b) Test accuracy {winner['test_accuracy']:.4f}: of the 10,000 "
            "held-out images the model",
            f"classifies {10000 - misclassified} correctly and "
            f"{misclassified} wrongly. Each image yields 10 softmax",
            "probabilities; the predicted digit is the argmax. The test set is "
            "never seen during",
            "training or epoch selection, so this is the generalization estimate.",
            "",
            "(c) " + shape_walk(results["Baseline CNN (ch. 5.1)"]["model"]).lstrip(),
            "",
        ]
    )
    (HERE / "part-3_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
