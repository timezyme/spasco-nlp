"""Part 1: baseline Reuters classifier and optimal-epoch selection.

Trains the reference 64-64 network for 20 epochs, picks the epoch with
the highest validation accuracy, retrains from scratch for exactly that
many epochs, and evaluates once on the test set. Writes
part-1_output.txt and training_history.png next to this script.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import keras

import reuters_common as rc

EPOCHS = 20
BATCH_SIZE = 512


def main() -> None:
    keras.utils.set_random_seed(rc.SEED)
    data = rc.load_reuters()

    # Exploratory run: train long enough to see overfitting set in.
    model = rc.build_baseline_model()
    history = rc.train_model(model, data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    best_epoch, best_val_acc = rc.best_epoch_from_history(history)

    rc.plot_training_history(
        history,
        best_epoch,
        "Reuters Baseline (64-64) - Training History",
        str(HERE / "training_history.png"),
    )

    # Final run: fresh model trained for exactly the selected epoch count,
    # then a single test-set evaluation.
    keras.utils.set_random_seed(rc.SEED)
    final_model = rc.build_baseline_model()
    rc.train_model(final_model, data, epochs=best_epoch, batch_size=BATCH_SIZE)
    test_pred = rc.predict_classes(final_model, data.x_test)

    majority = rc.majority_class_baseline(data.y_train_int, data.y_test_int)
    shuffled = rc.shuffled_labels_baseline(data.y_test_int)

    report = "\n".join(
        [
            "PART 1 - BASELINE MODEL AND OPTIMAL EPOCH SELECTION",
            "=" * 60,
            rc.versions_header(),
            "",
            "Configuration:",
            f"  Architecture: Dense 64-64, rmsprop, batch size {BATCH_SIZE}",
            f"  Splits: train {len(data.x_train)}, val {len(data.x_val)}, "
            f"test {len(data.x_test)}",
            "",
            "Epoch selection (on validation only):",
            f"  Explored epochs: {EPOCHS}",
            f"  Best epoch: {best_epoch} (val accuracy {best_val_acc:.4f})",
            "",
            "Test set (final model, single evaluation):",
            rc.metrics_block(data.y_test_int, test_pred),
            "Reference baselines:",
            f"  Majority class: {majority:.4f}",
            f"  Shuffled labels: {shuffled:.4f}",
            "",
        ]
    )
    (HERE / "part-1_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
