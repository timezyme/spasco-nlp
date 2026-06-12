#!/usr/bin/env python3
"""
Word2Vec Implementation for Shakespeare's Complete Works
This solution loads, tokenizes, and trains a Word2Vec model on Shakespeare's text
"""

import re
import numpy as np
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import random
import math
from datetime import datetime
import json


class Word2Vec:
    """Simple Word2Vec implementation using Skip-gram with negative sampling"""

    def __init__(self, vector_size=100, window=5, min_count=5,
                 negative_samples=5, learning_rate=0.025, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocabulary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_vectors = None
        self.context_vectors = None

    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        word_counts = Counter()
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1

        # Filter by minimum count
        vocab_words = [word for word, count in word_counts.items()
                      if count >= self.min_count]

        # Create word-to-index mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocabulary = word_counts

        # Initialize weight matrices
        vocab_size = len(self.word_to_idx)
        self.word_vectors = np.random.uniform(-0.5, 0.5,
                                             (vocab_size, self.vector_size)) / self.vector_size
        self.context_vectors = np.random.uniform(-0.5, 0.5,
                                                (vocab_size, self.vector_size)) / self.vector_size

        print(f"Vocabulary size: {vocab_size} words")

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def get_training_pairs(self, sentences):
        """Generate training pairs (center word, context word)"""
        pairs = []
        for sentence in sentences:
            sentence_indices = []
            for word in sentence:
                if word in self.word_to_idx:
                    sentence_indices.append(self.word_to_idx[word])

            for center_pos, center_idx in enumerate(sentence_indices):
                # Get context words within window
                context_start = max(0, center_pos - self.window)
                context_end = min(len(sentence_indices), center_pos + self.window + 1)

                for context_pos in range(context_start, context_end):
                    if context_pos != center_pos:
                        context_idx = sentence_indices[context_pos]
                        pairs.append((center_idx, context_idx))

        return pairs

    def get_negative_samples(self, context_idx):
        """Sample negative examples"""
        negatives = []
        vocab_size = len(self.word_to_idx)

        while len(negatives) < self.negative_samples:
            neg_idx = random.randint(0, vocab_size - 1)
            if neg_idx != context_idx and neg_idx not in negatives:
                negatives.append(neg_idx)

        return negatives

    def train_pair(self, center_idx, context_idx):
        """Train on a single center-context pair using negative sampling"""
        # Get vectors
        center_vec = self.word_vectors[center_idx]
        context_vec = self.context_vectors[context_idx]

        # Positive sample
        score = np.dot(center_vec, context_vec)
        prob = self.sigmoid(score)

        # Gradient for positive sample
        grad = (1 - prob) * self.learning_rate
        self.word_vectors[center_idx] += grad * context_vec
        self.context_vectors[context_idx] += grad * center_vec

        # Negative samples
        negative_indices = self.get_negative_samples(context_idx)
        for neg_idx in negative_indices:
            neg_vec = self.context_vectors[neg_idx]
            score = np.dot(center_vec, neg_vec)
            prob = self.sigmoid(score)

            # Gradient for negative sample
            grad = -prob * self.learning_rate
            self.word_vectors[center_idx] += grad * neg_vec
            self.context_vectors[neg_idx] += grad * center_vec

    def train(self, sentences):
        """Train the Word2Vec model"""
        print("Building vocabulary...")
        self.build_vocabulary(sentences)

        print("Generating training pairs...")
        pairs = self.get_training_pairs(sentences)
        print(f"Total training pairs: {len(pairs)}")

        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            random.shuffle(pairs)

            for i, (center_idx, context_idx) in enumerate(pairs):
                self.train_pair(center_idx, context_idx)

                if i % 10000 == 0:
                    progress = (i / len(pairs)) * 100
                    print(f"Epoch {epoch+1}/{self.epochs}: {progress:.1f}% complete", end='\r')

            # Decay learning rate
            self.learning_rate *= 0.95
            print(f"Epoch {epoch+1}/{self.epochs} completed. Learning rate: {self.learning_rate:.4f}")

    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.word_to_idx:
            return self.word_vectors[self.word_to_idx[word]]
        else:
            return None


def load_and_preprocess_text(filepath):
    """Load HTML file and extract text content"""
    print(f"Loading text from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')

    text_content = soup.get_text()

    # Basic preprocessing
    text_content = text_content.lower()

    # Remove extra whitespace
    text_content = re.sub(r'\s+', ' ', text_content)

    return text_content


def tokenize_text(text):
    """Tokenize text into sentences and words"""
    print("Tokenizing text...")

    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)

    # Tokenize each sentence
    tokenized_sentences = []
    for sentence in sentences:
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]+\b', sentence.lower())
        if len(words) > 1:  # Only keep sentences with at least 2 words
            tokenized_sentences.append(words)

    print(f"Total sentences: {len(tokenized_sentences)}")

    # Count total tokens
    total_tokens = sum(len(sent) for sent in tokenized_sentences)
    print(f"Total tokens: {total_tokens}")

    return tokenized_sentences


def display_word_vectors(model, words):
    """Display vector representations for specified words"""
    print("\n" + "="*60)
    print("Word Vector Representations")
    print("="*60)

    for word in words:
        vector = model.get_vector(word)
        if vector is not None:
            print(f"\nWord: '{word}'")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector (first 10 dimensions):")
            print(vector[:10])
            print(f"Vector norm: {np.linalg.norm(vector):.4f}")
            print(f"Min value: {vector.min():.4f}, Max value: {vector.max():.4f}")
        else:
            print(f"\nWord: '{word}' - Not found in vocabulary")


def generate_report(model, words, total_sentences, total_tokens):
    """Generate a detailed report with vector representations"""
    report = []
    report.append("="*80)
    report.append("WORD2VEC VECTOR REPRESENTATIONS REPORT")
    report.append("Shakespeare's Complete Works Analysis")
    report.append("="*80)
    report.append("")
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Dataset Statistics
    report.append("DATASET STATISTICS")
    report.append("-"*40)
    report.append(f"Total sentences analyzed: {total_sentences:,}")
    report.append(f"Total tokens: {total_tokens:,}")
    report.append(f"Vocabulary size: {len(model.word_to_idx):,} words")
    report.append(f"Training sentences used: 10,000 (subset for faster training)")
    report.append("")

    # Model Configuration
    report.append("MODEL CONFIGURATION")
    report.append("-"*40)
    report.append(f"Algorithm: Skip-gram with Negative Sampling")
    report.append(f"Vector dimensions: {model.vector_size}")
    report.append(f"Context window size: {model.window}")
    report.append(f"Minimum word count: {model.min_count}")
    report.append(f"Negative samples: {model.negative_samples}")
    report.append(f"Training epochs: {model.epochs}")
    report.append("")

    # Word Vector Representations
    report.append("="*80)
    report.append("WORD VECTOR REPRESENTATIONS")
    report.append("="*80)
    report.append("")

    vectors_data = {}

    for word in words:
        vector = model.get_vector(word)

        if vector is not None:
            report.append(f"WORD: '{word.upper()}'")
            report.append("-"*40)

            # Basic statistics
            report.append(f"Vector shape: {vector.shape}")
            report.append(f"Vector norm (L2): {np.linalg.norm(vector):.6f}")
            report.append(f"Mean value: {vector.mean():.6f}")
            report.append(f"Standard deviation: {vector.std():.6f}")
            report.append(f"Min value: {vector.min():.6f}")
            report.append(f"Max value: {vector.max():.6f}")
            report.append("")

            # Full vector representation
            report.append("Complete vector representation (100 dimensions):")
            report.append("")

            # Format vector in rows of 5 values each
            for i in range(0, len(vector), 5):
                dims = f"[{i:3d}-{min(i+4, len(vector)-1):3d}]: "
                values = " ".join([f"{v:8.5f}" for v in vector[i:i+5]])
                report.append(dims + values)

            report.append("")
            report.append("")

            # Store for JSON export
            vectors_data[word] = vector.tolist()

        else:
            report.append(f"WORD: '{word.upper()}'")
            report.append("-"*40)
            report.append("Word not found in vocabulary")
            report.append("")

    # Similarity Analysis
    report.append("="*80)
    report.append("COSINE SIMILARITY ANALYSIS")
    report.append("="*80)
    report.append("")

    word_pairs = [
        ("king", "queen"),
        ("king", "love"),
        ("king", "death"),
        ("queen", "love"),
        ("queen", "death"),
        ("love", "death")
    ]

    similarities = []

    for word1, word2 in word_pairs:
        vec1 = model.get_vector(word1)
        vec2 = model.get_vector(word2)

        if vec1 is not None and vec2 is not None:
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append((word1, word2, cosine_sim))
            report.append(f"'{word1}' <-> '{word2}': {cosine_sim:8.6f}")

    report.append("")

    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    report.append("Word pairs ranked by similarity:")
    report.append("-"*40)
    for word1, word2, sim in similarities:
        report.append(f"{sim:8.6f}: '{word1}' <-> '{word2}'")

    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    return "\n".join(report), vectors_data


def main():
    """Main execution function"""
    # File path
    filepath = '4/The Complete Works of William Shakespeare.html'

    # Step 1: Load and preprocess text
    text_content = load_and_preprocess_text(filepath)

    # Step 2: Tokenize the text
    tokenized_sentences = tokenize_text(text_content)

    # Store total statistics before limiting
    total_sentences = len(tokenized_sentences)
    total_tokens = sum(len(sent) for sent in tokenized_sentences)

    # Limit sentences for faster training (you can remove this for full training)
    print("\nUsing first 10000 sentences for training (for faster execution)...")
    tokenized_sentences = tokenized_sentences[:10000]

    # Step 3: Train Word2Vec model
    print("\nInitializing Word2Vec model...")
    model = Word2Vec(
        vector_size=100,
        window=5,
        min_count=5,
        negative_samples=5,
        learning_rate=0.025,
        epochs=3  # Reduced for faster execution
    )

    model.train(tokenized_sentences)

    # Step 4: Display vectors for target words
    target_words = ["king", "queen", "love", "death"]
    display_word_vectors(model, target_words)

    # Additional: Show some word relationships
    print("\n" + "="*60)
    print("Word Similarity Analysis")
    print("="*60)

    # Compute cosine similarity between word pairs
    word_pairs = [("king", "queen"), ("love", "death"), ("king", "love")]

    for word1, word2 in word_pairs:
        vec1 = model.get_vector(word1)
        vec2 = model.get_vector(word2)

        if vec1 is not None and vec2 is not None:
            # Cosine similarity
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(f"\nCosine similarity between '{word1}' and '{word2}': {cosine_sim:.4f}")
        else:
            missing = word1 if vec1 is None else word2
            print(f"\nCannot compute similarity - '{missing}' not in vocabulary")

    print("\n" + "="*60)
    print("Word2Vec training completed successfully!")
    print("="*60)

    # Step 5: Generate and save report
    print("\n" + "="*60)
    print("Generating detailed report...")
    print("="*60)

    report_text, vectors_data = generate_report(model, target_words, total_sentences, total_tokens)

    # Save text report
    report_filename = 'projects/4/part3/word2vec_report.txt'
    with open(report_filename, 'w') as f:
        f.write(report_text)
    print(f"Report saved to: {report_filename}")

    # Save JSON vectors
    json_filename = 'projects/4/part3/word_vectors.json'
    with open(json_filename, 'w') as f:
        json.dump(vectors_data, f, indent=2)
    print(f"Vector data saved to: {json_filename}")

    print("\nAll files generated successfully!")


if __name__ == "__main__":
    main()