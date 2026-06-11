"""Part 2: IMDB sentiment with a 200-word vocabulary.

Trains the Chollet ch. 3.5 Dense 16-16-1 classifier on multi-hot
vectors over only the 200 most frequent words: 20-epoch exploratory
run, epoch selection on validation loss, retrain on the full training
set, one test evaluation. Two reference points are measured under the
identical protocol instead of cited: the same architecture with the
full 10,000-word vocabulary, and TF-IDF + logistic regression capped
at the same 200 words. Also answers (c): what x_train[0] represents.
Writes part-2_output.txt and training_history.png next to this script.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import keras
import numpy as np

import imdb_common as ic

SMALL_VOCAB = 200
FULL_VOCAB = 10000
MAX_EPOCHS = 20
BATCH_SIZE = 512


def run_protocol(num_words: int) -> dict:
    """Explore on train/val, pick the epoch, retrain on all 25k, test once."""
    splits = ic.multi_hot_splits(num_words)

    keras.utils.set_random_seed(ic.SEED)
    model = ic.build_dense_model(num_words)
    history = ic.train_model(
        model,
        splits.x_train,
        splits.y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(splits.x_val, splits.y_val),
    )
    best_epoch, val_acc, _ = ic.best_epoch_from_history(history)

    keras.utils.set_random_seed(ic.SEED)
    final_model = ic.build_dense_model(num_words)
    ic.train_model(
        final_model, splits.x_all, splits.y_all, epochs=best_epoch, batch_size=BATCH_SIZE
    )
    test_loss, test_acc = final_model.evaluate(splits.x_test, splits.y_test, verbose=0)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "first_review": splits.x_all[0],
        "y_all": splits.y_all,
        "y_test": splits.y_test,
    }


def comparison_table(rows: dict[str, dict]) -> str:
    header = f"{'Model':<30} {'Vocab':>6} {'Best ep':>8} {'Val acc':>8} {'Test acc':>9}"
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        best = str(r.get("best_epoch", "-"))
        val = f"{r['val_accuracy']:.4f}" if "val_accuracy" in r else "-"
        lines.append(
            f"{name:<30} {r['vocab']:>6} {best:>8} {val:>8} {r['test_accuracy']:>9.4f}"
        )
    return "\n".join(lines)


def first_review_block(first_review: np.ndarray) -> str:
    """Requirement (c): what x_train[0] is, with measured counts."""
    present = int(first_review.sum())
    return "\n".join(
        [
            "(c) What x_train[0] represents:",
            f"  A single review as a {first_review.shape[0]}-dimensional multi-hot "
            "vector: position i is 1.0",
            "  if word index i occurs anywhere in the review, else 0.0. Order and",
            "  frequency are discarded (bag of words).",
            f"  Measured on the first training review: {present} of "
            f"{first_review.shape[0]} positions set.",
            "  Indices 0-2 are reserved markers (padding, sequence start, "
            "out-of-vocabulary),",
            "  so the count includes the start and OOV markers, not only real words.",
        ]
    )


def main() -> None:
    small = run_protocol(SMALL_VOCAB)
    full = run_protocol(FULL_VOCAB)
    classical = ic.tfidf_logreg_baseline(SMALL_VOCAB)
    majority = ic.majority_class_baseline(small["y_all"], small["y_test"])

    ic.plot_training_history(
        small["history"],
        small["best_epoch"],
        f"IMDB MLP (16-16), {SMALL_VOCAB}-word vocabulary - Training History",
        str(HERE / "training_history.png"),
    )

    rows = {
        f"MLP 16-16, {SMALL_VOCAB} words": {**small, "vocab": SMALL_VOCAB},
        f"MLP 16-16, {FULL_VOCAB} words": {**full, "vocab": FULL_VOCAB},
        "TF-IDF + LogisticRegression": {**classical, "vocab": SMALL_VOCAB},
        "Majority class": {"vocab": "-", "test_accuracy": majority},
    }
    retained = small["test_accuracy"] / full["test_accuracy"] * 100
    report = "\n".join(
        [
            "PART 2 - IMDB SENTIMENT WITH A 200-WORD VOCABULARY",
            "=" * 60,
            ic.versions_header(),
            "",
            "Protocol (identical for both vocabularies): explore for "
            f"{MAX_EPOCHS} epochs on the",
            f"15k/10k train/val split, select the epoch with minimum validation loss,",
            f"retrain from scratch on all 25k reviews, evaluate once on the test set.",
            f"Batch size {BATCH_SIZE}.",
            "",
            comparison_table(rows),
            "",
            f"With {SMALL_VOCAB / FULL_VOCAB:.0%} of the vocabulary the {SMALL_VOCAB}-word "
            f"model retains {retained:.1f}% of the",
            f"{FULL_VOCAB}-word test accuracy "
            f"({small['test_accuracy']:.4f} vs {full['test_accuracy']:.4f}).",
            "",
            first_review_block(small["first_review"]),
            "",
        ]
    )
    (HERE / "part-2_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
