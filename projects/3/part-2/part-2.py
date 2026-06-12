"""Part 2: bag-of-words model with an out-of-vocabulary token.

Tokenizes and POS-aware-lemmatizes two markdown documents, builds a
1-gram vocabulary from the larger one (index 0 reserved for <OOV>),
represents the smaller one as a {word index: frequency} dictionary, and
answers the spec questions: how many new words does the small text
contain, and which key marks them. Writes part-2_output.txt next to
this script.
"""

import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

HERE = Path(__file__).resolve().parent
LARGE_PATH = HERE / "sample-large.md"
SMALL_PATH = HERE / "sample-small.md"
OOV_INDEX = 0
TOP_DISPLAY = 20

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


def strip_markdown(text: str) -> str:
    text = re.sub(r"#+\s*", "", text)  # headers
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links: keep label
    return re.sub(r"[*_]", "", text)  # emphasis markers


def preprocess(text: str, lemmatizer: WordNetLemmatizer) -> list[str]:
    """Lowercase, strip markdown, tokenize, filter noise, lemmatize with POS."""
    tokens = word_tokenize(strip_markdown(text.lower()))
    tokens = [
        t
        for t in tokens
        if not all(c in string.punctuation for c in t)
        and not t.isdigit()
        and len(t) > 1
        and not t.startswith("http")
    ]
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, treebank_to_wordnet(tag)) for tok, tag in tagged]


def build_vocabulary(tokens: list[str]) -> dict[str, int]:
    """1-gram vocabulary, alphabetically indexed from 1; 0 is the <OOV> slot."""
    vocab = {"<OOV>": OOV_INDEX}
    for index, token in enumerate(sorted(set(tokens)), start=1):
        vocab[token] = index
    return vocab


def bow_representation(tokens: list[str], vocab: dict[str, int]) -> dict[int, int]:
    """{vocabulary index: frequency}; every unknown token counts under OOV_INDEX."""
    counts = Counter(vocab.get(token, OOV_INDEX) for token in tokens)
    return dict(counts)


def token_sample_block(tokens: list[str]) -> str:
    lines = ["First 50 processed tokens of the small text, ten per row:"]
    for start in range(0, 50, 10):
        lines.append(f"  {' '.join(tokens[start:start + 10])}")
    unique_sample = sorted(set(tokens))[:30]
    lines.append("")
    lines.append("First 30 unique lemmas of the small text:")
    for start in range(0, len(unique_sample), 6):
        lines.append(f"  {', '.join(unique_sample[start:start + 6])}")
    return "\n".join(lines)


def bow_table(bow: dict[int, int], vocab: dict[str, int]) -> str:
    index_to_word = {index: word for word, index in vocab.items()}
    by_frequency = sorted(bow.items(), key=lambda kv: (-kv[1], kv[0]))

    header = f"{'Index':>6}  {'Token':<22} {'Frequency':>9}"
    lines = [header, "-" * len(header)]
    for index, freq in by_frequency[:TOP_DISPLAY]:
        lines.append(f"{index:>6}  {index_to_word[index]:<22} {freq:>9}")
    singles = [(i, f) for i, f in by_frequency if f == 1][:10]
    lines.append(f"... plus {sum(1 for _, f in bow.items() if f == 1)} single-occurrence "
                 "entries, first 10:")
    for index, freq in singles:
        lines.append(f"{index:>6}  {index_to_word[index]:<22} {freq:>9}")
    return "\n".join(lines)


def versions_header() -> str:
    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: nltk {nltk.__version__}",
            "Deterministic run: no randomness involved",
        ]
    )


def main() -> None:
    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()

    large_tokens = preprocess(LARGE_PATH.read_text(encoding="utf-8"), lemmatizer)
    small_tokens = preprocess(SMALL_PATH.read_text(encoding="utf-8"), lemmatizer)

    vocab = build_vocabulary(large_tokens)
    bow = bow_representation(small_tokens, vocab)

    oov_occurrences = bow.get(OOV_INDEX, 0)
    oov_words = sorted({t for t in small_tokens if t not in vocab})
    known_unique = len(bow) - (1 if OOV_INDEX in bow else 0)
    coverage = known_unique / len(set(small_tokens)) * 100

    report = "\n".join(
        [
            "PART 2 - BAG OF WORDS WITH AN OOV TOKEN",
            "=" * 60,
            versions_header(),
            "",
            "(a) Tokenization and lemmatization (POS-aware, markdown stripped):",
            f"  Large text: {len(large_tokens)} tokens, "
            f"{len(set(large_tokens))} unique lemmas",
            f"  Small text: {len(small_tokens)} tokens, "
            f"{len(set(small_tokens))} unique lemmas",
            "",
            token_sample_block(small_tokens),
            "",
            "(b) Vocabulary from the large text and BoW for the small text:",
            f"  Vocabulary size: {len(vocab)} entries (including <OOV> at index "
            f"{OOV_INDEX})",
            f"  BoW dictionary for the small text: {len(bow)} distinct indices, "
            f"frequencies sum to {sum(bow.values())} (= its token count)",
            "",
            bow_table(bow, vocab),
            "",
            "(c) New words:",
            f"  The dictionary shows the new-word volume directly: index {OOV_INDEX} "
            f"carries {oov_occurrences} occurrences",
            f"  across {len(oov_words)} distinct words absent from the large-text "
            "vocabulary.",
            f"  The key for any new word is {OOV_INDEX} (the reserved <OOV> slot); "
            "individual new words are",
            "  indistinguishable inside the BoW, which is the cost of a closed "
            "vocabulary.",
            f"  Vocabulary coverage of the small text: {coverage:.1f}% of unique "
            "lemmas; sample OOV words:",
            f"  {', '.join(oov_words[:10])}",
            "",
        ]
    )
    (HERE / "part-2_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
