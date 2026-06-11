"""Part 2: improved architecture vs the baseline and a classical model.

Trains the 256-128-64 batch-norm network with early stopping, re-trains
the part-1 baseline for an in-run comparison, and runs a TF-IDF +
logistic regression classical baseline on the same splits. Reports a
comparison table plus per-class metrics and a confusion matrix for the
improved model. Writes part-2_output.txt, training_history.png, and
confusion_matrix.png next to this script.
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

import reuters_common as rc

MAX_EPOCHS = 30
BATCH_SIZE = 256


def train_and_evaluate(build_fn, data: rc.ReutersData) -> dict:
    """Train with early stopping (best val weights restored), evaluate once."""
    keras.utils.set_random_seed(rc.SEED)
    model = build_fn()
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max"
        ),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]
    start = time.perf_counter()
    history = rc.train_model(
        model, data, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks
    )
    fit_seconds = time.perf_counter() - start

    best_epoch, best_val_acc = rc.best_epoch_from_history(history)
    test_pred = rc.predict_classes(model, data.x_test)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "test_accuracy": accuracy_score(data.y_test_int, test_pred),
        "test_macro_f1": f1_score(
            data.y_test_int, test_pred, average="macro", zero_division=0
        ),
        "fit_seconds": fit_seconds,
        "test_pred": test_pred,
    }


def comparison_table(rows: dict[str, dict]) -> str:
    header = (
        f"{'Model':<28} {'Val acc':>8} {'Test acc':>9} {'Macro F1':>9} {'Fit (s)':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        lines.append(
            f"{name:<28} {r['val_accuracy']:>8.4f} {r['test_accuracy']:>9.4f} "
            f"{r['test_macro_f1']:>9.4f} {r['fit_seconds']:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    data = rc.load_reuters()

    improved = train_and_evaluate(rc.build_improved_model, data)
    baseline = train_and_evaluate(rc.build_baseline_model, data)
    classical = rc.tfidf_logreg_baseline()

    rc.plot_training_history(
        improved["history"],
        improved["best_epoch"],
        "Reuters Improved (256-128-64 + BN) - Training History",
        str(HERE / "training_history.png"),
    )
    rc.plot_confusion_matrix(
        data.y_test_int,
        improved["test_pred"],
        str(HERE / "confusion_matrix.png"),
        "Improved model - test confusion matrix (row-normalized)",
    )

    rows = {
        "Baseline MLP (64-64)": baseline,
        "Improved MLP (256-128-64)": improved,
        "TF-IDF + LogisticRegression": classical,
    }
    report = "\n".join(
        [
            "PART 2 - IMPROVED MODEL VS BASELINES",
            "=" * 60,
            rc.versions_header(),
            "",
            f"Training: max {MAX_EPOCHS} epochs, batch size {BATCH_SIZE}, "
            "early stopping on val accuracy (patience 5, best weights restored),",
            "ReduceLROnPlateau on val loss. Neural models share identical splits "
            "with the classical baseline.",
            "",
            comparison_table(rows),
            "",
            f"Improved model stopped at epoch {improved['best_epoch']} "
            f"(best validation accuracy {improved['val_accuracy']:.4f}).",
            "",
            "Improved model, test set detail:",
            rc.metrics_block(data.y_test_int, improved["test_pred"]),
            "",
        ]
    )
    (HERE / "part-2_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
