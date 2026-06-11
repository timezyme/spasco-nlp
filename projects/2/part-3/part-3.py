"""Part 3: IMDB sentiment with an Embedding + LSTM sequence model.

Pads every review to 500 word indices, embeds each index into a
200-dimensional vector (spec: the LSTM must see input of shape
(samples, 500, 200)), and trains LSTM(32) with early stopping on
validation loss. The best-validation weights are evaluated once on the
test set against the >= 75% accuracy target. Also answers (c): what
x_train[0] and x_train[0,0] represent. Writes part-3_output.txt and
training_history.png next to this script.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import keras
import numpy as np
from keras.callbacks import EarlyStopping

import imdb_common as ic

NUM_WORDS = 10000
MAXLEN = 500
EMBEDDING_DIM = 200  # spec: LSTM input shape (samples, 500, 200)
LSTM_UNITS = 32
MAX_EPOCHS = 20
BATCH_SIZE = 128
TARGET_ACCURACY = 0.75


def representation_block(splits: ic.ImdbSplits) -> str:
    """Requirement (c): what x_train[0] and x_train[0,0] are, with measured values."""
    raw_first = ic.load_imdb_raw(NUM_WORDS)[0][0][0]
    first = splits.x_all[0]
    pad_count = int(np.sum(first == 0))
    return "\n".join(
        [
            "(c) Data representation:",
            f"  x_train[0] is one complete review as a ({MAXLEN},) integer sequence:",
            "  each element is a word's rank index in the frequency-ordered",
            f"  {NUM_WORDS}-word vocabulary. The Embedding layer maps every index to a",
            f"  learned {EMBEDDING_DIM}-dim vector, giving the LSTM a "
            f"({MAXLEN}, {EMBEDDING_DIM}) sequence per review.",
            f"  The raw review has {len(raw_first)} words, so pre-padding fills the "
            f"first {pad_count}",
            f"  positions with 0.",
            f"  x_train[0,0] = {int(first[0])}: index 0 is the padding token, not a "
            "word - the",
            "  review is shorter than 500 words, and pad_sequences pads at the front.",
        ]
    )


def main() -> None:
    splits = ic.padded_splits(NUM_WORDS, MAXLEN)

    keras.utils.set_random_seed(ic.SEED)
    model = ic.build_lstm_model(NUM_WORDS, MAXLEN, EMBEDDING_DIM, LSTM_UNITS)
    history = ic.train_model(
        model,
        splits.x_train,
        splits.y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(splits.x_val, splits.y_val),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ],
    )
    best_epoch, best_val_acc, _ = ic.best_epoch_from_history(history)
    epochs_run = len(history.history["loss"])

    # EarlyStopping restored the best-validation-loss weights; this is
    # the single test evaluation.
    test_loss, test_acc = model.evaluate(splits.x_test, splits.y_test, verbose=0)

    ic.plot_training_history(
        history,
        best_epoch,
        "IMDB Embedding(200) + LSTM(32) - Training History",
        str(HERE / "training_history.png"),
    )

    target = (
        f"Target accuracy >= {TARGET_ACCURACY:.0%}: "
        f"{'achieved' if test_acc >= TARGET_ACCURACY else 'NOT met'}"
    )
    report = "\n".join(
        [
            "PART 3 - IMDB SENTIMENT WITH EMBEDDING + LSTM",
            "=" * 60,
            ic.versions_header(),
            "",
            "Configuration:",
            f"  Vocabulary {NUM_WORDS}, sequences pre-padded to {MAXLEN}",
            f"  Embedding({NUM_WORDS}, {EMBEDDING_DIM}) -> LSTM({LSTM_UNITS}) -> "
            "Dense(1, sigmoid)",
            f"  {model.count_params()} parameters, rmsprop, binary cross-entropy",
            f"  Training: max {MAX_EPOCHS} epochs, batch {BATCH_SIZE}, 15k/10k "
            "train/val split,",
            "  early stopping on val loss (patience 3, best weights restored)",
            "",
            "Epoch selection (on validation only):",
            f"  Epochs run: {epochs_run}",
            f"  Best epoch: {best_epoch} (val accuracy {best_val_acc:.4f})",
            "",
            "Test set (best-validation weights, single evaluation):",
            f"  Accuracy: {test_acc:.4f}",
            f"  Loss:     {test_loss:.4f}",
            f"  {target}",
            "",
            representation_block(splits),
            "",
        ]
    )
    (HERE / "part-3_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
