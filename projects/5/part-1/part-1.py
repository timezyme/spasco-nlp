"""Part 1: character-level embeddings and what transfers to a new text.

Builds a character vocabulary from a large text (deep-sea prose), trains a
seeded next-character model in Keras whose first layer is a character
Embedding, and then applies those learned vectors to a small text that
contains symbols the large text never used. Every spec question is answered
with a measurement rather than an assertion: the embedding geometry is read
out as cosine nearest neighbors and a within-class vs across-class
similarity gap, the transfer is quantified as shared-character coverage, and
the out-of-vocabulary behavior is probed against the live Keras layer (an
out-of-range index raises, so the reserved <UNK> slot is our design, not a
Keras Embedding default). Writes part-1_output.txt and embedding_pca.png
next to this script; saves no model.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras import layers, models, optimizers
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
LARGE_PATH = HERE / "large.txt"
SMALL_PATH = HERE / "small.txt"
SEED = 42
CONTEXT = 8  # characters of context used to predict the next character
EMBED_DIM = 24
HIDDEN = 128
EPOCHS = 60
BATCH = 128
UNK = "<UNK>"  # reserved at index 0 for any character absent from the large text
VOWELS = set("aeiou")
NEIGHBOR_PROBES = ["a", "e", "t", "n", " ", ".", ","]


def build_vocab(text: str) -> tuple[list[str], dict, dict]:
    """Vocabulary from the large text only, with <UNK> reserved at index 0."""
    vocab = [UNK] + sorted(set(text))
    char_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_char = {i: ch for i, ch in enumerate(vocab)}
    return vocab, char_to_int, int_to_char


def encode(text: str, char_to_int: dict) -> list[int]:
    unk = char_to_int[UNK]
    return [char_to_int.get(ch, unk) for ch in text]


def make_sequences(encoded: list[int], context: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of `context` characters predicting the next one."""
    x, y = [], []
    for i in range(len(encoded) - context):
        x.append(encoded[i : i + context])
        y.append(encoded[i + context])
    return np.array(x), np.array(y)


def build_model(vocab_size: int) -> models.Model:
    return models.Sequential(
        [
            layers.Input(shape=(CONTEXT,)),
            layers.Embedding(vocab_size, EMBED_DIM, name="char_embedding"),
            layers.Flatten(),
            layers.Dense(HIDDEN, activation="relu"),
            layers.Dense(vocab_size, activation="softmax"),
        ]
    )


def char_class(ch: str) -> str:
    if ch == UNK:
        return "unk"
    if ch == " ":
        return "space"
    if ch == "\n":
        return "newline"
    if ch in VOWELS:
        return "vowel"
    if ch.isalpha():
        return "consonant"
    return "punct"


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    return normed @ normed.T


def nearest_neighbors(
    sims: np.ndarray, int_to_char: dict, query_idx: int, topn: int = 4
) -> list[tuple[str, float]]:
    order = np.argsort(sims[query_idx])[::-1]
    out = []
    for j in order:
        if j == query_idx or int_to_char[j] == UNK:
            continue
        out.append((int_to_char[j], float(sims[query_idx][j])))
        if len(out) == topn:
            break
    return out


def class_similarity_gap(sims: np.ndarray, classes: list[str]) -> tuple[float, float]:
    """Mean cosine within the vowel set vs vowel-to-consonant, a measured test
    of whether the embedding groups characters of a kind."""
    vowels = [i for i, c in enumerate(classes) if c == "vowel"]
    consonants = [i for i, c in enumerate(classes) if c == "consonant"]
    within = [sims[i][j] for i in vowels for j in vowels if i < j]
    across = [sims[i][j] for i in vowels for j in consonants]
    return float(np.mean(within)), float(np.mean(across))


def display_char(ch: str) -> str:
    return {"\n": "\\n", " ": "space", "\t": "\\t"}.get(ch, ch)


def small_mapping_block(small_text: str, encoded: list[int], int_to_char: dict, n: int) -> str:
    lines = [f"First {n} characters of the small text, lowercased, as char -> int:"]
    row = []
    for ch, idx in list(zip(small_text, encoded))[:n]:
        tag = "*UNK*" if int_to_char[idx] == UNK else ""
        row.append(f"{display_char(ch)}:{idx}{tag}")
        if len(row) == 6:
            lines.append("  " + "  ".join(row))
            row = []
    if row:
        lines.append("  " + "  ".join(row))
    return "\n".join(lines)


def probe_oov(model: models.Model, vocab_size: int) -> str:
    """Look up an index one past the vocabulary against the live layer and
    report exactly what Keras does, rather than describing it from memory."""
    emb = model.get_layer("char_embedding")
    try:
        emb(np.array([[vocab_size]]))
        return "an out-of-range index returned a value without error (unexpected)"
    except Exception as exc:  # noqa: BLE001 - we want the concrete type name
        return f"an out-of-range index raises {type(exc).__name__}"


def plot_pca(
    vectors: np.ndarray, labels: list[str], classes: list[str], save_path: Path
) -> None:
    coords = PCA(n_components=2, random_state=SEED).fit_transform(vectors)
    palette = {
        "vowel": "#d62728",
        "consonant": "#1f77b4",
        "punct": "#2ca02c",
        "space": "#9467bd",
        "newline": "#8c564b",
        "unk": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in palette:
        pts = [i for i, c in enumerate(classes) if c == cls]
        if pts:
            ax.scatter(coords[pts, 0], coords[pts, 1], s=45, c=palette[cls], label=cls)
    for (x, y), ch in zip(coords, labels):
        ax.annotate(display_char(ch), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_title("Character embeddings by class (PCA of the trained Embedding layer)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def versions_header() -> str:
    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: keras {keras.__version__}, scikit-learn {sklearn.__version__}",
            f"Random seed: {SEED} (keras.utils.set_random_seed)",
        ]
    )


def main() -> None:
    keras.utils.set_random_seed(SEED)

    large_text = LARGE_PATH.read_text(encoding="utf-8").lower()
    small_text = SMALL_PATH.read_text(encoding="utf-8").lower()

    vocab, char_to_int, int_to_char = build_vocab(large_text)
    vocab_size = len(vocab)
    large_encoded = encode(large_text, char_to_int)
    small_encoded = encode(small_text, char_to_int)

    x_train, y_train = make_sequences(large_encoded, CONTEXT)
    model = build_model(vocab_size)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.1,
        verbose=0,
    )
    val_acc = float(history.history["val_accuracy"][-1])

    # Majority-class baseline for next-character prediction: always predict the
    # single most frequent target. Beating it shows the model (hence the
    # embedding) captured real sequential structure.
    counts = np.bincount(y_train, minlength=vocab_size)
    majority_acc = float(counts.max() / counts.sum())
    most_common_char = int_to_char[int(counts.argmax())]

    embeddings = model.get_layer("char_embedding").get_weights()[0]
    small_embeddings = embeddings[small_encoded]
    classes = [char_class(ch) for ch in vocab]
    sims = cosine_matrix(embeddings)
    within_vowel, vowel_to_consonant = class_similarity_gap(sims, classes)

    neighbor_lines = []
    for ch in NEIGHBOR_PROBES:
        if ch in char_to_int:
            pairs = nearest_neighbors(sims, int_to_char, char_to_int[ch])
            shown = ", ".join(f"'{display_char(w)}' ({s:+.2f})" for w, s in pairs)
            neighbor_lines.append(f"  '{display_char(ch)}' -> {shown}")

    small_chars = set(small_text)
    large_chars = set(large_text)
    new_chars = sorted(small_chars - large_chars)
    covered = sum(1 for ch in small_text if ch in large_chars)
    oov_total = len(small_text) - covered

    plot_pca(embeddings, vocab, classes, HERE / "embedding_pca.png")
    oov_behavior = probe_oov(model, vocab_size)

    sample_idx = char_to_int["e"]
    report = "\n".join(
        [
            "PART 1 - CHARACTER EMBEDDINGS AND TRANSFER TO A NEW TEXT",
            "=" * 60,
            versions_header(),
            "",
            f"Large text: {LARGE_PATH.name} ({len(large_text)} chars). "
            f"Small text: {SMALL_PATH.name} ({len(small_text)} chars).",
            "",
            "(a) Preprocessing: both texts lowercased; the character vocabulary is",
            f"built from the large text only, with {UNK} reserved at index 0.",
            f"  Vocabulary size: {vocab_size} ({vocab_size - 1} distinct characters + {UNK}).",
            small_mapping_block(small_text, small_encoded, int_to_char, 36),
            "",
            "(b) A next-character model (Embedding -> Flatten -> Dense -> softmax)",
            f"trained on {len(x_train):,} sliding windows of {CONTEXT} characters.",
            f"  Final validation accuracy: {val_acc:.3f} "
            f"vs majority baseline {majority_acc:.3f} "
            f"(always predicting '{display_char(most_common_char)}').",
            f"  Trained embedding matrix: {embeddings.shape} "
            f"(one {EMBED_DIM}-d row per vocabulary character).",
            f"  Applying it to the small text gives an array of shape "
            f"{small_embeddings.shape}",
            f"  (one row per character). First two rows, '{display_char(small_text[0])}' "
            f"and '{display_char(small_text[1])}':",
            f"    {np.array2string(small_embeddings[0][:6], precision=3, floatmode='fixed')} ...",
            f"    {np.array2string(small_embeddings[1][:6], precision=3, floatmode='fixed')} ...",
            f"  Example learned vector, 'e' (index {sample_idx}), first 6 dims:",
            f"    {np.array2string(embeddings[sample_idx][:6], precision=3, floatmode='fixed')} ...",
            "",
            "(c) What transfers from the large text to the small text's vectors.",
            "  Two things, and both are measured here rather than asserted:",
            "  1. The shared character vocabulary. Every column the small text can",
            "     use was defined by the large text; the small text adds none of",
            "     its own. The vectors are read straight from the trained matrix.",
            "  2. The embedding weights, i.e. the geometry. The model places",
            "     characters that share predictive context near each other.",
            "     Measured cosine nearest neighbors:",
            *neighbor_lines,
            f"     Within-vowel mean cosine {within_vowel:+.2f} vs "
            f"vowel-to-consonant {vowel_to_consonant:+.2f}; the gap of "
            f"{within_vowel - vowel_to_consonant:+.2f} is the",
            "     sense in which 'a kind of character' forms a group. See",
            "     embedding_pca.png for the full layout colored by class.",
            "  Significance: a shared character in the small text inherits a vector",
            "  shaped by every occurrence in the large text, for free and without",
            "  retraining. That is the transfer; it is only as meaningful as the",
            f"  model is non-trivial, which the {val_acc:.3f} vs {majority_acc:.3f} "
            "gap establishes.",
            "",
            "(d) New characters in the small text and Keras OOV behavior.",
            f"  {len(new_chars)} distinct characters appear in the small text but not",
            f"  the large text: {' '.join(display_char(c) for c in new_chars)}",
            f"  They account for {oov_total} of {len(small_text)} small-text characters "
            f"({100 * oov_total / len(small_text):.1f}%), all mapped to {UNK} (index 0).",
            "  Keras default, probed against the live layer in this run:",
            f"    keras.layers.Embedding has no OOV slot; {oov_behavior}",
            "    (index one past the vocabulary). So nothing about <UNK> is a Keras",
            "    Embedding default; reserving index 0 is our own design, and without",
            "    it these characters would crash the lookup, not fall back quietly.",
            "  For contrast, keras.layers.TextVectorization does reserve slots by",
            "  default: index 0 for padding and index 1 for '[UNK]'. Different layer,",
            "  different contract; the embedding lookup itself forgives nothing.",
            "",
        ]
    )
    (HERE / "part-1_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
