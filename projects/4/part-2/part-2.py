"""Part 2: IMDB sentiment with TF-IDF n-grams, targeting >= 0.90 test accuracy.

Loads the Keras IMDB data with num_words=10000 (unchanged, per spec),
decodes the integer sequences back to text, strips the leftover HTML
token "br", and represents reviews as sublinear TF-IDF over word
n-grams. Protocol, all selection on a held-out validation split: the
n-gram range is chosen by a (1,1)/(1,2)/(1,3) ablation with a C-tuned
logistic regression, then three models (tuned logistic regression,
single-unit Keras logistic, 512-unit MLP) compete on the chosen
representation. Every candidate refits on the full 25k training set at
its selected epoch / best C and is evaluated on the test set exactly
once, with the winner declared on validation accuracy before any test
evaluation. Writes part-2_output.txt and training_history.png next to
this script; saves no model files.
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
from keras import layers, models, optimizers, regularizers
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
NUM_WORDS = 10000
MAX_FEATURES = 40000
NGRAM_CANDIDATES = ((1, 1), (1, 2), (1, 3))
VAL_FRACTION = 0.1
MAX_EPOCHS = 20
BATCH_SIZE = 512
LOGREG_C_GRID = (1.0, 2.0, 4.0)
INDEX_FROM = 3  # keras imdb convention: 0=pad, 1=start, 2=oov


def decode_reviews(sequences) -> list[str]:
    """Map integer sequences back to words; markers and "br" are dropped.

    Indices 0/1/2 (pad/start/oov) have no word and fall out of the join;
    "br" is the residue of <br /> tags in the source HTML and carries no
    sentiment, yet it was the single highest-weighted TF-IDF feature in
    the original version of this project.
    """
    word_by_index = {
        rank + INDEX_FROM: word for word, rank in imdb.get_word_index().items()
    }
    return [
        " ".join(
            w for i in seq if (w := word_by_index.get(i)) is not None and w != "br"
        )
        for seq in sequences
    ]


def vectorize(train_texts, ngram_range, max_features):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return vectorizer, vectorizer.fit_transform(train_texts)


def split(x_all, labels):
    return train_test_split(
        x_all, labels, test_size=VAL_FRACTION, random_state=SEED, stratify=labels
    )


def evaluate_range(train_texts, labels, ngram_range) -> dict:
    """Vectorize one n-gram range; tune logreg C on the validation split."""
    vectorizer, x_all = vectorize(train_texts, ngram_range, MAX_FEATURES)
    x_tr, x_val, y_tr, y_val = split(x_all, labels)
    scored = []
    for c in LOGREG_C_GRID:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y_tr)
        scored.append((accuracy_score(y_val, clf.predict(x_val)), c))
    val_acc, best_c = max(scored)
    return {
        "label": f"({ngram_range[0]},{ngram_range[1]})",
        "vectorizer": vectorizer,
        "x_all": x_all,
        "n_features": len(vectorizer.vocabulary_),
        "best_c": best_c,
        "val_accuracy": float(val_acc),
    }


class SparseBatches(keras.utils.PyDataset):
    """Feeds a scipy CSR matrix to Keras, densifying one batch at a time.

    The full 25k x 40k float32 design matrix would be 4 GB dense; a
    512-row batch is 82 MB.
    """

    def __init__(self, x_csr, y, batch_size, shuffle=False, seed=SEED, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x_csr, np.asarray(y, dtype="float32")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(x_csr.shape[0])
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        rows = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self.x[rows].toarray(), self.y[rows]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def build_keras_logistic(input_dim: int) -> keras.Model:
    model = models.Sequential(
        [keras.Input(shape=(input_dim,)), layers.Dense(1, activation="sigmoid")]
    )
    model.compile(
        optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def build_mlp(input_dim: int) -> keras.Model:
    model = models.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(
                512, activation="relu", kernel_regularizer=regularizers.l2(5e-4)
            ),
            layers.Dropout(0.45),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.RMSprop(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def explore_keras(build_fn, x_tr, y_tr, x_val, y_val) -> dict:
    """Fit with early stopping; epoch selection on validation accuracy."""
    keras.utils.set_random_seed(SEED)
    model = build_fn(x_tr.shape[1])
    history = model.fit(
        SparseBatches(x_tr, y_tr, BATCH_SIZE, shuffle=True),
        validation_data=SparseBatches(x_val, y_val, BATCH_SIZE),
        epochs=MAX_EPOCHS,
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy", mode="max", patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    return {
        "history": history.history,
        "best_epoch": best_epoch,
        "val_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
    }


def final_fit_keras(build_fn, x_full, y_full, x_test, y_test, epochs: int) -> float:
    """Refit on the full training set for the selected epochs; one test eval."""
    keras.utils.set_random_seed(SEED)
    model = build_fn(x_full.shape[1])
    model.fit(
        SparseBatches(x_full, y_full, BATCH_SIZE, shuffle=True),
        epochs=epochs,
        verbose=0,
    )
    _, test_acc = model.evaluate(
        SparseBatches(x_test, y_test, BATCH_SIZE), verbose=0
    )
    return float(test_acc)


def plot_history(history: dict, best_epoch: int, title: str, save_path: Path) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in ((ax1, "accuracy", "Accuracy"), (ax2, "loss", "Loss")):
        ax.plot(epochs, history[metric], "bo-", label=f"Training {label.lower()}", markersize=4)
        ax.plot(
            epochs, history[f"val_{metric}"], "r^-",
            label=f"Validation {label.lower()}", markersize=4,
        )
        ax.axvline(
            x=best_epoch, color="green", linestyle="--",
            label=f"Selected epoch ({best_epoch})",
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
    start = time.perf_counter()
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=NUM_WORDS
    )
    train_texts = decode_reviews(train_data)
    test_texts = decode_reviews(test_data)

    # Stage 1 (validation only): choose the n-gram range.
    ranges = [
        evaluate_range(train_texts, train_labels, ngram_range)
        for ngram_range in NGRAM_CANDIDATES
    ]
    chosen = max(ranges, key=lambda r: r["val_accuracy"])
    x_train_full = chosen["x_all"]
    x_test = chosen["vectorizer"].transform(test_texts)
    best_c = chosen["best_c"]
    x_tr, x_val, y_tr, y_val = split(x_train_full, train_labels)

    # Stage 2 (validation only): model comparison on the chosen representation.
    logistic = explore_keras(build_keras_logistic, x_tr, y_tr, x_val, y_val)
    mlp = explore_keras(build_mlp, x_tr, y_tr, x_val, y_val)
    candidates = {
        f"LogisticRegression (C={best_c:g})": {"val_accuracy": chosen["val_accuracy"]},
        "Keras logistic (Dense 1)": logistic,
        "Keras MLP 512 (drop 0.45, L2)": mlp,
    }
    winner_name = max(candidates, key=lambda k: candidates[k]["val_accuracy"])

    # Stage 3: full-train refits; each candidate touches the test set once.
    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
    clf.fit(x_train_full, train_labels)
    tests = {
        f"LogisticRegression (C={best_c:g})": float(
            accuracy_score(test_labels, clf.predict(x_test))
        ),
        "Keras logistic (Dense 1)": final_fit_keras(
            build_keras_logistic, x_train_full, train_labels, x_test, test_labels,
            logistic["best_epoch"],
        ),
        "Keras MLP 512 (drop 0.45, L2)": final_fit_keras(
            build_mlp, x_train_full, train_labels, x_test, test_labels,
            mlp["best_epoch"],
        ),
    }
    winner_test = tests[winner_name]

    keras_plotted = mlp if mlp["val_accuracy"] >= logistic["val_accuracy"] else logistic
    plot_history(
        keras_plotted["history"],
        keras_plotted["best_epoch"],
        f"IMDB TF-IDF {chosen['label']} - Keras MLP training",
        HERE / "training_history.png",
    )

    feature_names = chosen["vectorizer"].get_feature_names_out()
    mean_scores = np.asarray(x_train_full.mean(axis=0)).ravel()
    top = ", ".join(feature_names[i] for i in np.argsort(mean_scores)[::-1][:10])

    abl_header = f"{'n-grams':<10} {'Features':>9} {'Best C':>7} {'Val acc':>8}"
    abl = [abl_header, "-" * len(abl_header)]
    abl += [
        f"{r['label']:<10} {r['n_features']:>9,} {r['best_c']:>7g} "
        f"{r['val_accuracy']:>8.4f}" + ("  <- chosen" if r is chosen else "")
        for r in ranges
    ]
    bigram_gain = 100 * (ranges[1]["val_accuracy"] - ranges[0]["val_accuracy"])
    trigram_gain = 100 * (ranges[2]["val_accuracy"] - ranges[1]["val_accuracy"])

    header = f"{'Model':<32} {'Best ep':>8} {'Val acc':>8} {'Test acc':>9}"
    table = [header, "-" * len(header)]
    for name, r in candidates.items():
        ep = str(r.get("best_epoch", "-"))
        table.append(
            f"{name:<32} {ep:>8} {r['val_accuracy']:>8.4f} {tests[name]:>9.4f}"
        )

    report = "\n".join(
        [
            "PART 2 - IMDB SENTIMENT WITH TF-IDF N-GRAMS",
            "=" * 60,
            versions_header(),
            "",
            f"Data: imdb.load_data(num_words={NUM_WORDS}) unchanged per spec; "
            "reviews decoded",
            'back to text, pad/start/oov markers and the HTML token "br" removed.',
            f"Representation: sublinear TF-IDF, max_features={MAX_FEATURES:,}, "
            "min_df=2, max_df=0.9,",
            "L2-normalized rows, sparse end to end (Keras reads CSR batches",
            "through a PyDataset). All selection on a 22.5k/2.5k stratified",
            "validation split; every candidate then refits on the full 25k train",
            "and touches the test set exactly once.",
            "",
            "N-gram selection (validation accuracy, C-tuned logistic regression):",
            *abl,
            "",
            f"Model comparison on the {chosen['label']} representation:",
            *table,
            "",
            f"Winner on validation: {winner_name}",
            f"TEST ACCURACY: {winner_test:.4f} "
            f"({'>= 0.90 goal met' if winner_test >= 0.90 else 'goal 0.90 missed'})",
            "",
            "Why this n-gram choice works: the big step is unigrams to bigrams",
            f"(+{bigram_gain:.2f} points of validation accuracy), because negation "
            "and intensity",
            "are two-word phenomena (not good, very best) that no bag of single",
            f"words can encode. Extending to trigrams adds {trigram_gain:+.2f} "
            "points at the same",
            "40k budget: min_df=2 already drops the one-off trigrams, so the",
            "surviving ones (mostly negated bigrams with a pivot word) are cheap",
            "to include. The original version of this project concluded trigrams",
            "hurt; with the br token removed and a convex, fully optimized",
            "classifier doing the comparison, the measured ordering reverses,",
            "which is why representation choices get re-tested here instead of",
            "inherited.",
            "",
            f"Top mean-TF-IDF training features (after cleanup): {top}",
            f"Total runtime: {time.perf_counter() - start:.0f}s",
            "",
        ]
    )
    (HERE / "part-2_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
