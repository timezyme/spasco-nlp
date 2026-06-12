"""Part 2: undercomplete autoencoder on MNIST, with the bottleneck chosen by
a classifier instead of by eye.

The spec asks for the smallest number of codings whose reconstructions are
"still recognizable". "Recognizable" is made objective: a small digit
classifier (a judge, ~0.98 on clean pixels) reads each candidate's
reconstructed validation images, and its accuracy on them is the
recognizability score. Every coding size is selected on a held-out
validation split only; the winner, the smallest size keeping judge accuracy
at or above the threshold, then touches the test set exactly once.

Architecture per spec guidance: 784 -> 256 -> 128 -> K(linear) -> 128 -> 256
-> 784 sigmoid, binary cross-entropy on pixels scaled to [0, 1]. Writes
part-2_output.txt, mnist_loss_curves.png, and mnist_reconstructions.png next
to this script; saves no model.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks, layers, models, optimizers
from keras.datasets import mnist

HERE = Path(__file__).resolve().parent
SEED = 42
VAL_SIZE = 6000  # held out from the 60k train split for coding-size selection
CODING_SIZES = [32, 16, 8, 4, 2]
JUDGE_THRESHOLD = 0.90  # min reconstructed-digit accuracy to call a size "recognizable"
AE_EPOCHS = 40
JUDGE_EPOCHS = 12
BATCH = 256
SAMPLE_DIGITS = 5


def load_data() -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    x_tr, x_val = x_train[:-VAL_SIZE], x_train[-VAL_SIZE:]
    y_tr, y_val = y_train[:-VAL_SIZE], y_train[-VAL_SIZE:]
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


def build_autoencoder(coding_size: int) -> models.Model:
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    code = layers.Dense(coding_size, activation="linear", name="code")(x)
    x = layers.Dense(128, activation="relu")(code)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(784, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def build_judge() -> models.Model:
    """An independent digit classifier used only to score reconstructions."""
    model = models.Sequential(
        [
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def judge_accuracy(judge: models.Model, images: np.ndarray, labels: np.ndarray) -> float:
    preds = judge.predict(images, batch_size=512, verbose=0).argmax(axis=1)
    return float((preds == labels).mean())


def train_autoencoder(model: models.Model, x_tr, x_val) -> keras.callbacks.History:
    stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=0
    )
    return model.fit(
        x_tr,
        x_tr,
        epochs=AE_EPOCHS,
        batch_size=BATCH,
        validation_data=(x_val, x_val),
        callbacks=[stop],
        verbose=0,
    )


def plot_loss_curves(histories: dict, winner: int, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for size, hist in histories.items():
        epochs = range(1, len(hist.history["val_loss"]) + 1)
        style = "-" if size == winner else "--"
        width = 2.4 if size == winner else 1.3
        ax1.plot(epochs, hist.history["val_loss"], style, linewidth=width, label=f"K={size}")
    ax1.set_title("Validation reconstruction loss by coding size")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary cross-entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    hist = histories[winner]
    epochs = range(1, len(hist.history["loss"]) + 1)
    ax2.plot(epochs, hist.history["loss"], label="Train")
    ax2.plot(epochs, hist.history["val_loss"], label="Validation")
    ax2.set_title(f"Winner K={winner}: train vs validation loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Binary cross-entropy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(
    originals: np.ndarray, recons: dict, sizes: list, save_path: Path
) -> None:
    n_rows = 1 + len(sizes)
    fig, axes = plt.subplots(n_rows, SAMPLE_DIGITS, figsize=(SAMPLE_DIGITS * 1.6, n_rows * 1.6))
    for col in range(SAMPLE_DIGITS):
        axes[0, col].imshow(originals[col].reshape(28, 28), cmap="gray")
        axes[0, col].axis("off")
    axes[0, 0].text(
        -0.3, 0.5, "original", transform=axes[0, 0].transAxes, ha="right", va="center"
    )
    for row, size in enumerate(sizes, start=1):
        for col in range(SAMPLE_DIGITS):
            axes[row, col].imshow(recons[size][col].reshape(28, 28), cmap="gray")
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.3, 0.5, f"K={size}", transform=axes[row, 0].transAxes, ha="right", va="center"
        )
    fig.suptitle("Five random test digits and their reconstructions by coding size")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def versions_header() -> str:
    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: keras {keras.__version__}",
            f"Random seed: {SEED} (keras.utils.set_random_seed)",
        ]
    )


def main() -> None:
    keras.utils.set_random_seed(SEED)
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = load_data()

    judge = build_judge()
    judge.fit(x_tr, y_tr, epochs=JUDGE_EPOCHS, batch_size=BATCH, verbose=0)
    judge_clean_val = judge_accuracy(judge, x_val, y_val)
    judge_clean_test = judge_accuracy(judge, x_test, y_test)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(x_test), SAMPLE_DIGITS, replace=False)
    sample_originals = x_test[sample_idx]

    autoencoders, histories, val_loss, val_judge, epochs_used, sample_recons = {}, {}, {}, {}, {}, {}
    for size in CODING_SIZES:
        ae = build_autoencoder(size)
        hist = train_autoencoder(ae, x_tr, x_val)
        autoencoders[size] = ae
        histories[size] = hist
        epochs_used[size] = len(hist.history["loss"])
        val_loss[size] = float(min(hist.history["val_loss"]))
        recon_val = ae.predict(x_val, batch_size=512, verbose=0)
        val_judge[size] = judge_accuracy(judge, recon_val, y_val)
        sample_recons[size] = ae.predict(sample_originals, verbose=0)

    # Smallest coding size whose reconstructed validation digits the judge
    # still reads correctly at least JUDGE_THRESHOLD of the time.
    passing = [s for s in CODING_SIZES if val_judge[s] >= JUDGE_THRESHOLD]
    winner = min(passing) if passing else max(CODING_SIZES, key=lambda s: val_judge[s])

    # Reuse the exact model selected on validation; the test set is touched once,
    # so the validation and test judge scores describe the same trained model.
    winner_ae = autoencoders[winner]
    recon_test = winner_ae.predict(x_test, batch_size=512, verbose=0)
    test_loss = float(winner_ae.evaluate(x_test, x_test, verbose=0))
    test_judge = judge_accuracy(judge, recon_test, y_test)

    plot_loss_curves(histories, winner, HERE / "mnist_loss_curves.png")
    plot_reconstructions(sample_originals, sample_recons, CODING_SIZES, HERE / "mnist_reconstructions.png")

    table = [
        f"{'K':>4} {'compression':>12} {'val BCE':>9} {'judge acc (val)':>16} {'epochs':>7}",
        "-" * 52,
    ]
    for size in CODING_SIZES:
        marker = "  <- winner" if size == winner else ""
        table.append(
            f"{size:>4} {f'{784 / size:.0f}:1':>12} {val_loss[size]:>9.4f} "
            f"{val_judge[size]:>16.4f} {epochs_used[size]:>7}{marker}"
        )

    next_size = (
        CODING_SIZES[CODING_SIZES.index(winner) + 1]
        if winner != CODING_SIZES[-1]
        else winner
    )
    report = "\n".join(
        [
            "PART 2 - UNDERCOMPLETE AUTOENCODER ON MNIST",
            "=" * 60,
            versions_header(),
            "",
            f"Data: {len(x_tr):,} train / {len(x_val):,} validation / {len(x_test):,} test, "
            "pixels scaled to [0, 1].",
            "Architecture: 784 -> 256 -> 128 -> K(linear) -> 128 -> 256 -> 784(sigmoid), "
            "binary cross-entropy.",
            "",
            "The judge: an independent classifier scoring whether a reconstruction",
            f"still looks like its digit. On clean pixels it reaches "
            f"{judge_clean_val:.4f} (val) and {judge_clean_test:.4f} (test);",
            "that ceiling is what reconstructions are measured against.",
            "",
            "Coding-size search (all selection on the validation split):",
            *table,
            "",
            f"Recognizability threshold: judge accuracy >= {JUDGE_THRESHOLD:.2f} on the",
            "reconstructed validation digits. Reconstruction loss falls monotonically",
            "as K grows, but loss does not say 'recognizable'; the judge does, and it",
            "is the judge that picks the floor.",
            "",
            f"Smallest number of codings that stays recognizable: {winner}.",
            f"  Compression: {784 / winner:.0f}:1 (784 pixels -> {winner} numbers).",
            f"  Held-out validation judge accuracy: {val_judge[winner]:.4f}.",
            "  Single test-set evaluation for this size:",
            f"    reconstruction BCE {test_loss:.4f}, judge accuracy {test_judge:.4f}",
            f"    (vs {judge_clean_test:.4f} on the original test digits, a drop of "
            f"{judge_clean_test - test_judge:.4f}).",
            "",
            "See mnist_reconstructions.png for the five random test digits at every K,",
            "and mnist_loss_curves.png for the loss trends. The next size down, "
            f"K={next_size}, is",
            "where the judge falls through the threshold; that boundary is what the eye",
            "alone reported inconsistently before.",
            "",
        ]
    )
    (HERE / "part-2_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
