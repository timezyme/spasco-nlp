"""Part 3: optimizer comparison for the improved Reuters classifier.

Trains the same 256-128-64 network under five optimizer families, two
hyperparameter settings each, ranks configurations by best validation
accuracy, and evaluates ONLY the winner on the test set (one test
evaluation total; the test set plays no part in selection). Writes
part-3_output.txt and optimizer_comparison.png next to this script.

This script is the deliberate exception to the one-configuration-per-run
convention: the deliverable here IS the cross-configuration table and
the comparison plots, which need every configuration's epoch history in
one place. To iterate on a single configuration, set RUN_ONLY to its
label (e.g. "Adam lr=1e-4"); that run prints validation metrics only
and leaves the results file and plots untouched.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adamax, AdamW, Nadam, RMSprop
from sklearn.metrics import accuracy_score, f1_score

import reuters_common as rc

MAX_EPOCHS = 25
BATCH_SIZE = 256
RUN_ONLY: str | None = None

# (family, label, optimizer factory) - factories so each run gets a fresh optimizer
CONFIGS = [
    ("RMSprop", "RMSprop lr=1e-3", lambda: RMSprop(learning_rate=1e-3)),
    ("RMSprop", "RMSprop lr=1e-3 rho=0.95", lambda: RMSprop(learning_rate=1e-3, rho=0.95)),
    ("Adam", "Adam lr=1e-3", lambda: Adam(learning_rate=1e-3)),
    ("Adam", "Adam lr=1e-4", lambda: Adam(learning_rate=1e-4)),
    ("Adamax", "Adamax lr=2e-3", lambda: Adamax(learning_rate=2e-3)),
    ("Adamax", "Adamax lr=1e-3", lambda: Adamax(learning_rate=1e-3)),
    ("AdamW", "AdamW lr=1e-3 wd=1e-2", lambda: AdamW(learning_rate=1e-3, weight_decay=1e-2)),
    ("AdamW", "AdamW lr=1e-3 wd=1e-3", lambda: AdamW(learning_rate=1e-3, weight_decay=1e-3)),
    ("Nadam", "Nadam lr=1e-3", lambda: Nadam(learning_rate=1e-3)),
    ("Nadam", "Nadam lr=2e-3", lambda: Nadam(learning_rate=2e-3)),
]


def run_config(family: str, label: str, make_optimizer, data: rc.ReutersData) -> dict:
    """Train one configuration; keep validation metrics and the fitted model."""
    keras.utils.set_random_seed(rc.SEED)
    model = rc.build_improved_model(optimizer=make_optimizer())
    stopper = EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max"
    )
    history = rc.train_model(
        model, data, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[stopper]
    )
    best_epoch, best_val_acc = rc.best_epoch_from_history(history)
    return {
        "family": family,
        "label": label,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "val_curve": history.history["val_accuracy"],
        "model": model,
    }


def results_table(results: list[dict]) -> str:
    header = f"{'Rank':<5} {'Configuration':<26} {'Best val acc':>12} {'Best epoch':>11}"
    lines = [header, "-" * len(header)]
    for rank, r in enumerate(results, 1):
        lines.append(
            f"{rank:<5} {r['label']:<26} {r['val_accuracy']:>12.4f} {r['best_epoch']:>11}"
        )
    return "\n".join(lines)


def plot_comparison(results: list[dict], save_path: str) -> None:
    """Three panels: best-per-family curves, ranked bars, Adam learning rates."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    families = sorted({r["family"] for r in results})
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    family_color = dict(zip(families, colors))

    for family in families:
        best = max(
            (r for r in results if r["family"] == family),
            key=lambda r: r["val_accuracy"],
        )
        epochs = range(1, len(best["val_curve"]) + 1)
        ax1.plot(
            epochs, best["val_curve"], label=best["label"],
            color=family_color[family], alpha=0.8,
        )
    ax1.set_title("Validation Accuracy - Best Configuration per Family")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation accuracy")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ranked = sorted(results, key=lambda r: r["val_accuracy"])
    bars = ax2.barh(
        [r["label"] for r in ranked],
        [r["val_accuracy"] for r in ranked],
        color=[family_color[r["family"]] for r in ranked],
    )
    ax2.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax2.set_xlim(0.5, 0.9)
    ax2.set_title("All Configurations Ranked by Best Validation Accuracy")
    ax2.set_xlabel("Best validation accuracy")
    ax2.tick_params(axis="y", labelsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    for r in results:
        if r["family"] == "Adam":
            epochs = range(1, len(r["val_curve"]) + 1)
            ax3.plot(epochs, r["val_curve"], label=r["label"], alpha=0.8)
    ax3.set_title("Adam - Learning Rate Comparison")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Validation accuracy")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Optimizer Comparison - Reuters Improved Model", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = rc.load_reuters()

    configs = CONFIGS
    if RUN_ONLY is not None:
        configs = [c for c in CONFIGS if c[1] == RUN_ONLY]
        if not configs:
            raise SystemExit(f"No configuration labeled {RUN_ONLY!r}")

    results = []
    for family, label, make_optimizer in configs:
        result = run_config(family, label, make_optimizer, data)
        results.append(result)
        print(
            f"{label:<26} val acc {result['val_accuracy']:.4f} "
            f"at epoch {result['best_epoch']}"
        )

    if RUN_ONLY is not None:
        return  # iteration mode: no ranking, no test evaluation, no files

    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    winner = results[0]
    test_pred = rc.predict_classes(winner["model"], data.x_test)
    test_acc = accuracy_score(data.y_test_int, test_pred)
    test_f1 = f1_score(data.y_test_int, test_pred, average="macro", zero_division=0)

    plot_comparison(results, str(HERE / "optimizer_comparison.png"))

    report = "\n".join(
        [
            "PART 3 - OPTIMIZER COMPARISON",
            "=" * 60,
            rc.versions_header(),
            "",
            f"Protocol: {len(results)} configurations, identical model "
            f"(256-128-64 + BN), identical seed,",
            f"max {MAX_EPOCHS} epochs, batch size {BATCH_SIZE}, early stopping on "
            "val accuracy (patience 5).",
            "Selection uses validation accuracy only; the single winner gets the "
            "one and only test evaluation.",
            "",
            results_table(results),
            "",
            f"Winner: {winner['label']}",
            f"  Test accuracy: {test_acc:.4f}",
            f"  Test macro F1: {test_f1:.4f}",
            "",
        ]
    )
    (HERE / "part-3_output.txt").write_text(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
