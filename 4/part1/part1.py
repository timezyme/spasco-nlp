"""
Assignment 4: Problem 1 - TF-IDF Vectorization with NLTK Preprocessing

This module implements TF-IDF vectorization with NLTK preprocessing,
demonstrating vocabulary transfer and OOV handling in text processing.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Compile regex pattern once for efficiency
ALPHA_PATTERN = re.compile(r'[^a-zA-Z\s]+')

# Download required NLTK data
for resource in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

def preprocess_text(text, lemmatizer):
    """
    Apply tokenization and lemmatization to text.

    Args:
        text (str): Input text to preprocess
        lemmatizer: NLTK WordNetLemmatizer instance

    Returns:
        str: Preprocessed text with tokens lemmatized and joined
    """
    # Convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = ALPHA_PATTERN.sub(' ', text)  # Use pre-compiled pattern

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

def main():
    print("="*80)
    print("Assignment 4 - Problem 1: TF-IDF Vectorization with NLTK Preprocessing")
    print("="*80)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Part (a): Apply word tokenization and lemmatization
    print("\n" + "="*80)
    print("PART (a): Word Tokenization and Lemmatization on BOTH Texts")
    print("="*80)

    # Read the large text dataset with error handling
    try:
        with open('large.md', 'r', encoding='utf-8') as f:
            large_text = f.read()
    except FileNotFoundError:
        print("Error: large.md file not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error reading large.md: {e}")
        return

    # Read the smaller text with new words
    try:
        with open('small.md', 'r', encoding='utf-8') as f:
            small_text = f.read()
    except FileNotFoundError:
        print("Error: small.md file not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error reading small.md: {e}")
        return

    print(f"\nOriginal large text length: {len(large_text)} characters")
    print(f"Original small text length: {len(small_text)} characters")

    # Preprocess both texts
    processed_large_text = preprocess_text(large_text, lemmatizer)
    processed_small_text = preprocess_text(small_text, lemmatizer)

    # Display samples of the processed LARGE text
    large_words = processed_large_text.split()
    print("\n--- Samples from processed LARGE text ---")
    print(f"Total tokens after processing: {len(large_words)}")
    print(f"First 30 tokens: {' '.join(large_words[:30])}")
    print(f"Middle 30 tokens: {' '.join(large_words[len(large_words)//2:len(large_words)//2+30])}")
    print(f"Last 30 tokens: {' '.join(large_words[-30:])}")

    # Display samples of the processed SMALLER text
    small_words = processed_small_text.split()
    print("\n--- Samples from processed SMALLER text ---")
    print(f"Total tokens after processing: {len(small_words)}")
    print(f"First 30 tokens: {' '.join(small_words[:30])}")
    print(f"Middle 30 tokens: {' '.join(small_words[len(small_words)//2:len(small_words)//2+30])}")
    print(f"Last 30 tokens: {' '.join(small_words[-30:])}")

    # Additional analysis: Vocabulary overlap
    large_vocab = set(large_words)
    small_vocab = set(small_words)
    common_words = large_vocab & small_vocab
    new_words_in_small = small_vocab - large_vocab
    print(f"\nVocabulary Analysis:")
    print(f"Large text unique tokens: {len(large_vocab)}")
    print(f"Small text unique tokens: {len(small_vocab)}")
    print(f"Common tokens: {len(common_words)}")
    print(f"New tokens in small text (not in large): {len(new_words_in_small)}")
    if new_words_in_small:
        print(f"Examples of new words: {list(new_words_in_small)[:10]}")

    # Part (b): Apply TF-IDF vectorization using Scikit-learn
    print("\n" + "="*80)
    print("PART (b): TF-IDF Vectorization")
    print("="*80)

    # Create TF-IDF vectorizer
    # Note: Since we need multiple documents for IDF calculation,
    # we split the large text into chunks
    # Using fixed-size chunks for consistency with original implementation
    words = processed_large_text.split()
    chunk_size = 100  # Fixed chunk size for reproducibility
    large_chunks = [' '.join(words[i:i+chunk_size])
                   for i in range(0, len(words), chunk_size)]

    print(f"\nCreated {len(large_chunks)} document chunks for IDF calculation")

    tfidf_vectorizer = TfidfVectorizer(
        preprocessor=lambda x: x,  # Already preprocessed
        tokenizer=lambda x: x.split(),  # Simple split
        max_features=500,
        min_df=1,
        max_df=0.9,
    )

    # Fit the TF-IDF vectorizer on the large text dataset
    tfidf_vectorizer.fit(large_chunks)
    print(f"\nVocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    # Apply the trained TF-IDF vectorizer to the smaller text
    small_text_tfidf = tfidf_vectorizer.transform([processed_small_text])

    # Display TF-IDF representation
    print(f"\nTF-IDF representation shape: {small_text_tfidf.shape}")
    print(f"Number of non-zero features: {small_text_tfidf.nnz}")

    # Get top TF-IDF features for the smaller text
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = small_text_tfidf.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[::-1][:10]

    print("\n--- Top 10 TF-IDF features in smaller text ---")
    for i, idx in enumerate(top_indices, 1):
        if tfidf_scores[idx] > 0:
            print(f"{i}. '{feature_names[idx]}': {tfidf_scores[idx]:.3f}")

    # Verify L2 normalization
    l2_norm = np.linalg.norm(tfidf_scores)
    print(f"\nL2 norm of TF-IDF vector: {l2_norm:.6f} (should be ~1.0)")


    # Part (c): Information transfer analysis
    print("\n" + "="*80)
    print("PART (c): Information Transfer from Large to Small Text")
    print("="*80)
    print("""
Information transferred from large text to small text TF-IDF representation:

1. **Vocabulary**: Only terms from the large text can have non-zero TF-IDF values.
2. **IDF weights**: Document frequency statistics computed from large text corpus.
3. **Feature space**: The 500 selected features based on large text term frequencies.
4. **Normalization**: L2 normalization scheme from the training corpus.
""")

    # Part (d): Handling of new words
    print("\n" + "="*80)
    print("PART (d): How Scikit-learn's TF-IDF Handles New Words")
    print("="*80)
    print("""
How Scikit-learn's TF-IDF handles new words:

• New words are silently ignored (no error or warning)
• They receive TF-IDF value of 0 (no contribution to vector)
• No <OOV> token is used to represent unknown words
• This causes information loss for out-of-vocabulary terms
""")

if __name__ == "__main__":
    main()