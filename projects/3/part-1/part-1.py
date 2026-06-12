"""Part 1: NLTK vs spaCy preprocessing on one small document.

Runs both libraries over the same song lyrics (song.txt) and compares
what each produces: sentence segmentation, word tokenization, stemming
(Porter and Snowball, NLTK only), POS-aware lemmatization, stop-word
removal, top-frequency words, and named entities (spaCy only). Writes
part-1_output.txt and comparison.png next to this script. Deterministic;
no model training involved.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import spacy
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

HERE = Path(__file__).resolve().parent
SONG_PATH = HERE / "song.txt"

# Words from the lyrics whose stems and lemmas separate the algorithms.
SAMPLE_WORDS = ["like", "caught", "making", "escape", "lady", "was", "sleeping"]
TOP_N = 10

NLTK_DATA = {
    "tokenizers": ["punkt", "punkt_tab"],
    "taggers": ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"],
    "corpora": ["wordnet", "omw-1.4", "stopwords"],
}


def ensure_nltk_data() -> None:
    """Download required NLTK packages if missing (correct category paths)."""
    for category, packages in NLTK_DATA.items():
        for package in packages:
            try:
                nltk.data.find(f"{category}/{package}")
            except LookupError:
                nltk.download(package, quiet=True)


def treebank_to_wordnet(tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS the lemmatizer expects."""
    mapping = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return mapping.get(tag[:1], wordnet.NOUN)


def nltk_pipeline(text: str) -> dict:
    """Tokenize, tag, stem, and lemmatize with NLTK; count stop-word removal."""
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    alpha = [t.lower() for t in tokens if t.isalpha()]

    tagged = pos_tag(alpha)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(tok, treebank_to_wordnet(tag)) for tok, tag in tagged]

    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    stop_set = set(stopwords.words("english"))
    content = [t for t in alpha if t not in stop_set]

    return {
        "sentences": sentences,
        "tokens": tokens,
        "alpha": alpha,
        "tagged": dict(tagged),
        "lemmas_by_token": dict(zip([t for t, _ in tagged], lemmas)),
        "lemmas": lemmas,
        "porter": {w: porter.stem(w) for w in set(alpha)},
        "snowball": {w: snowball.stem(w) for w in set(alpha)},
        "stopwords_removed": len(alpha) - len(content),
        "top_content": Counter(content).most_common(TOP_N),
    }


def spacy_pipeline(text: str) -> dict:
    """Tokenize, tag, and lemmatize with spaCy; collect sentences and entities."""
    doc = spacy.load("en_core_web_sm")(text)
    alpha_tokens = [t for t in doc if t.is_alpha]
    content = [t.lemma_.lower() for t in alpha_tokens if not t.is_stop]

    return {
        "doc": doc,
        "sentences": list(doc.sents),
        "tokens": [t.text for t in doc],
        "alpha": [t.text.lower() for t in alpha_tokens],
        "lemmas_by_token": {t.text.lower(): t.lemma_.lower() for t in alpha_tokens},
        "pos_by_token": {t.text.lower(): t.pos_ for t in alpha_tokens},
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "stopwords_removed": sum(1 for t in alpha_tokens if t.is_stop),
        "top_content": Counter(content).most_common(TOP_N),
    }


def lemma_agreement(nl: dict, sp: dict) -> tuple[int, int]:
    """(identical lemmas, comparable words) over the shared alpha vocabulary."""
    shared = set(nl["lemmas_by_token"]) & set(sp["lemmas_by_token"])
    same = sum(1 for w in shared if nl["lemmas_by_token"][w] == sp["lemmas_by_token"][w])
    return same, len(shared)


def word_comparison_table(nl: dict, sp: dict) -> str:
    header = (
        f"{'Word':<12} {'Porter':<12} {'Snowball':<12} "
        f"{'NLTK lemma':<12} {'spaCy lemma':<12}"
    )
    lines = [header, "-" * len(header)]
    for word in SAMPLE_WORDS:
        if word not in nl["lemmas_by_token"]:
            continue  # not every candidate survives tokenization of the lyrics
        lines.append(
            f"{word:<12} {nl['porter'].get(word, '-'):<12} "
            f"{nl['snowball'].get(word, '-'):<12} "
            f"{nl['lemmas_by_token'].get(word, '-'):<12} "
            f"{sp['lemmas_by_token'].get(word, '-'):<12}"
        )
    return "\n".join(lines)


def counts_table(nl: dict, sp: dict) -> str:
    rows = [
        ("Sentences detected", len(nl["sentences"]), len(sp["sentences"])),
        ("Word tokens (raw)", len(nl["tokens"]), len(sp["tokens"])),
        ("Alphabetic tokens", len(nl["alpha"]), len(sp["alpha"])),
        ("Unique lemmas", len(set(nl["lemmas"])), len(set(sp["lemmas_by_token"].values()))),
        ("Stop-word tokens removed", nl["stopwords_removed"], sp["stopwords_removed"]),
        ("Named entities", "-", len(sp["entities"])),
    ]
    header = f"{'Measure':<26} {'NLTK':>8} {'spaCy':>8}"
    lines = [header, "-" * len(header)]
    for name, a, b in rows:
        lines.append(f"{name:<26} {a!s:>8} {b!s:>8}")
    return "\n".join(lines)


def frequency_block(nl: dict, sp: dict) -> str:
    lines = [f"{'NLTK top content words':<28} {'spaCy top content lemmas':<28}"]
    lines.append("-" * 56)
    for (w1, c1), (w2, c2) in zip(nl["top_content"], sp["top_content"]):
        lines.append(f"{w1 + ' (' + str(c1) + ')':<28} {w2 + ' (' + str(c2) + ')':<28}")
    return "\n".join(lines)


def plot_comparison(nl: dict, sp: dict, save_path: Path) -> None:
    """Two panels: pipeline counts side by side, and NLTK top content words."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    measures = ["Sentences", "Alpha tokens", "Unique lemmas", "Stop words removed"]
    nltk_vals = [
        len(nl["sentences"]),
        len(nl["alpha"]),
        len(set(nl["lemmas"])),
        nl["stopwords_removed"],
    ]
    spacy_vals = [
        len(sp["sentences"]),
        len(sp["alpha"]),
        len(set(sp["lemmas_by_token"].values())),
        sp["stopwords_removed"],
    ]
    x = range(len(measures))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], nltk_vals, width, label="NLTK")
    ax1.bar([i + width / 2 for i in x], spacy_vals, width, label="spaCy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(measures, rotation=15)
    ax1.set_title("Pipeline measures")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    words = [w for w, _ in nl["top_content"]]
    counts = [c for _, c in nl["top_content"]]
    ax2.barh(words[::-1], counts[::-1])
    ax2.set_title(f"Top {TOP_N} content words (NLTK, stop words removed)")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("NLTK vs spaCy on one song lyric")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def versions_header() -> str:
    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: nltk {nltk.__version__}, spacy {spacy.__version__} "
            "(en_core_web_sm)",
            "Deterministic run: no randomness involved",
        ]
    )


def main() -> None:
    ensure_nltk_data()
    text = SONG_PATH.read_text(encoding="utf-8")

    nl = nltk_pipeline(text)
    sp = spacy_pipeline(text)
    same, shared = lemma_agreement(nl, sp)

    plot_comparison(nl, sp, HERE / "comparison.png")

    entities = ", ".join(f"{t} [{label}]" for t, label in sp["entities"][:12])
    report = "\n".join(
        [
            "PART 1 - NLTK VS SPACY PREPROCESSING COMPARISON",
            "=" * 60,
            versions_header(),
            "",
            f"Document: {SONG_PATH.name} ({len(text)} characters of song lyrics)",
            "",
            counts_table(nl, sp),
            "",
            "Sentence segmentation: the lyrics contain almost no terminal",
            "punctuation, so NLTK's punkt finds "
            f"{len(nl['sentences'])} sentence(s); spaCy's parser uses",
            f"linebreaks and syntax and finds {len(sp['sentences'])}.",
            "",
            "Stems and lemmas for selected words:",
            word_comparison_table(nl, sp),
            "",
            f"Lemma agreement on the {shared} shared alphabetic words: "
            f"{same} identical ({same / shared:.0%}).",
            "Disagreements are mostly irregular forms where spaCy's tagger",
            "picks the verb reading (caught -> catch, was -> be).",
            "",
            frequency_block(nl, sp),
            "",
            f"Named entities (spaCy only, first 12 of {len(sp['entities'])}):",
            f"  {entities}",
            "",
        ]
    )
    (HERE / "part-1_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
