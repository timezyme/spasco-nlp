"""Part 1: TF-IDF vocabulary transfer from a large text to a small one.

Tokenizes and POS-aware-lemmatizes two research-paper texts with NLTK,
fits a scikit-learn TfidfVectorizer on the large one (chunked into
pseudo-documents so IDF has a corpus to count over), and applies it to
the small one. Answers the spec questions with measured numbers: what
information transfers with the vectorizer, and what happens to the
small text's new words. Writes part-1_output.txt and tfidf_features.png
next to this script. Deterministic; no model training involved.
"""

import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import sklearn
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
LARGE_PATH = HERE / "large.md"
SMALL_PATH = HERE / "small.md"
CHUNK_SIZE = 100  # tokens per pseudo-document for IDF
MAX_FEATURES = 500
TOP_N = 10

NON_ALPHA = re.compile(r"[^a-z\s]+")

NLTK_DATA = {
    "tokenizers": ["punkt", "punkt_tab"],
    "taggers": ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"],
    "corpora": ["wordnet", "omw-1.4"],
}


def ensure_nltk_data() -> None:
    for category, packages in NLTK_DATA.items():
        for package in packages:
            try:
                nltk.data.find(f"{category}/{package}")
            except LookupError:
                nltk.download(package, quiet=True)


def treebank_to_wordnet(tag: str) -> str:
    mapping = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return mapping.get(tag[:1], wordnet.NOUN)


def preprocess(text: str, lemmatizer: WordNetLemmatizer) -> list[str]:
    """(a) Lowercase, drop non-alphabetic characters, tokenize, lemmatize."""
    tokens = word_tokenize(NON_ALPHA.sub(" ", text.lower()))
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, treebank_to_wordnet(tag)) for tok, tag in tagged]


def chunk_tokens(tokens: list[str], size: int) -> list[str]:
    """Split one long token stream into fixed-size pseudo-documents."""
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens), size)]


def fit_and_transform(
    chunks: list[str], small_text: str, stop_words: str | None
) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, min_df=1, max_df=0.9, stop_words=stop_words
    )
    vectorizer.fit(chunks)
    return vectorizer, vectorizer.transform([small_text]).toarray()[0]


def top_features(vectorizer: TfidfVectorizer, scores: np.ndarray) -> list[tuple[str, float]]:
    names = vectorizer.get_feature_names_out()
    order = np.argsort(scores)[::-1][:TOP_N]
    return [(names[i], float(scores[i])) for i in order if scores[i] > 0]


def feature_table(plain: list[tuple[str, float]], filtered: list[tuple[str, float]]) -> str:
    header = f"{'No stop-word filter':<28} {'stop_words=english':<28}"
    lines = [header, "-" * len(header)]
    for (w1, s1), (w2, s2) in zip(plain, filtered):
        lines.append(f"{w1 + f' ({s1:.3f})':<28} {w2 + f' ({s2:.3f})':<28}")
    return "\n".join(lines)


def sample_block(tokens: list[str], label: str) -> str:
    lines = [f"First 30 processed tokens of the {label} text:"]
    for start in (0, 10, 20):
        lines.append(f"  {' '.join(tokens[start:start + 10])}")
    return "\n".join(lines)


def plot_features(
    plain: list[tuple[str, float]], filtered: list[tuple[str, float]], save_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, feats, title in (
        (axes[0], plain, "No stop-word filter"),
        (axes[1], filtered, "stop_words='english'"),
    ):
        words = [w for w, _ in feats][::-1]
        scores = [s for _, s in feats][::-1]
        ax.barh(words, scores)
        ax.set_title(f"Top TF-IDF features of the small text ({title})")
        ax.set_xlabel("TF-IDF score")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def versions_header() -> str:
    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: nltk {nltk.__version__}, scikit-learn {sklearn.__version__}",
            "Deterministic run: no randomness involved",
        ]
    )


def main() -> None:
    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()

    large_raw = LARGE_PATH.read_text(encoding="utf-8")
    small_raw = SMALL_PATH.read_text(encoding="utf-8")
    large_tokens = preprocess(large_raw, lemmatizer)
    small_tokens = preprocess(small_raw, lemmatizer)
    small_text = " ".join(small_tokens)

    chunks = chunk_tokens(large_tokens, CHUNK_SIZE)
    vectorizer, scores = fit_and_transform(chunks, small_text, stop_words=None)
    vec_filtered, scores_filtered = fit_and_transform(chunks, small_text, "english")

    new_words = sorted(set(small_tokens) - set(large_tokens))
    in_feature_space = [w for w in new_words if w in vectorizer.vocabulary_]
    plain_top = top_features(vectorizer, scores)
    filtered_top = top_features(vec_filtered, scores_filtered)
    plot_features(plain_top, filtered_top, HERE / "tfidf_features.png")

    report = "\n".join(
        [
            "PART 1 - TF-IDF VOCABULARY TRANSFER (LARGE -> SMALL TEXT)",
            "=" * 60,
            versions_header(),
            "",
            f"Documents: {LARGE_PATH.name} ({len(large_raw)} chars) -> vectorizer; "
            f"{SMALL_PATH.name} ({len(small_raw)} chars) -> transformed",
            "",
            "(a) NLTK preprocessing (lowercase, alphabetic only, word_tokenize,",
            "POS-aware WordNet lemmatization) applied to both texts:",
            f"  Large text: {len(large_tokens)} tokens, "
            f"{len(set(large_tokens))} unique lemmas",
            f"  Small text: {len(small_tokens)} tokens, "
            f"{len(set(small_tokens))} unique lemmas",
            "",
            sample_block(small_tokens, "small"),
            "",
            "(b) TF-IDF fit on the large text, applied to the small text.",
            f"  The large text becomes {len(chunks)} pseudo-documents of "
            f"{CHUNK_SIZE} tokens each, because IDF",
            "  needs document frequencies; a single document would make every "
            "IDF identical.",
            f"  Vocabulary size: {len(vectorizer.vocabulary_)} (max_features="
            f"{MAX_FEATURES})",
            f"  Small-text representation: shape (1, {scores.size}), "
            f"{int(np.count_nonzero(scores))} non-zero entries,",
            f"  L2 norm {np.linalg.norm(scores):.6f} (sklearn normalizes each "
            "vector to unit length).",
            "",
            "  Top features with and without stop-word filtering:",
            feature_table(plain_top, filtered_top),
            "",
            "  Without filtering, function words (in, we, to) dominate even under",
            "  IDF, because 23 short chunks of one paper give them only mildly",
            "  deflated weights; filtering moves content terms (training, loss,",
            "  modality) to the top. Both vectorizers transfer identically; only",
            "  the vocabulary differs.",
            "",
            "(c) What transferred from the large text into the small text's vector:",
            "  1. The vocabulary: all "
            f"{len(vectorizer.vocabulary_)} feature columns come from the large "
            "text; the small",
            "     text cannot introduce columns of its own.",
            "  2. The IDF weights: each column's idf_ was computed from document",
            "     frequencies in the large-text chunks, so the large text decides",
            "     which words count as distinctive.",
            "  3. The preprocessing contract: the same lowercase/token/lemma",
            "     pipeline must be applied, or tokens miss the vocabulary.",
            "  4. The normalization scheme: L2 unit length, inherited from the",
            "     fitted transformer's configuration.",
            "",
            "(d) New words in the small text:",
            f"  {len(new_words)} unique lemmas of the small text never occur in "
            "the large text.",
            f"  {len(in_feature_space)} of them made it into the vectorizer's "
            "vocabulary; scikit-learn drops the rest",
            "  silently. No error, no warning, no <OOV> slot; transform() simply",
            "  skips tokens absent from vocabulary_, so every new word contributes",
            "  zero to the vector. Compare project 3 part 2, where a hand-rolled",
            "  bag-of-words reserved index 0 and counted the same kind of words",
            "  explicitly (145 OOV occurrences there). sklearn's choice keeps the",
            "  feature space fixed but makes vocabulary mismatch invisible unless",
            "  you measure it, as done here.",
            f"  Sample new words: {', '.join(new_words[:10])}",
            "",
        ]
    )
    (HERE / "part-1_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
