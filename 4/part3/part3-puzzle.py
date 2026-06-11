#!/usr/bin/env python3
"""
Word2Vec Analogy Solver
Solves: "king" is to "queen" as "boy" is to ???
Using vector arithmetic: boy + queen - king
"""

import re
import numpy as np
from collections import Counter
from bs4 import BeautifulSoup
import random
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

    def find_closest_word(self, target_vector, exclude_words=None):
        """Find the word whose vector is closest to the target vector"""
        if exclude_words is None:
            exclude_words = []

        best_word = None
        best_similarity = -1
        best_angle = 180

        target_norm = np.linalg.norm(target_vector)

        for word in self.word_to_idx:
            if word in exclude_words:
                continue

            word_vec = self.get_vector(word)
            word_norm = np.linalg.norm(word_vec)

            # Cosine similarity
            cosine_sim = np.dot(target_vector, word_vec) / (target_norm * word_norm)

            # Angle in degrees
            angle_radians = np.arccos(np.clip(cosine_sim, -1, 1))
            angle_degrees = np.degrees(angle_radians)

            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                best_word = word
                best_angle = angle_degrees

        return best_word, best_similarity, best_angle

    def solve_analogy(self, word_a, word_b, word_c, top_n=5):
        """
        Solve analogy: word_a is to word_b as word_c is to ???
        Formula: result = word_c + word_b - word_a
        """
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        vec_c = self.get_vector(word_c)

        if vec_a is None or vec_b is None or vec_c is None:
            return None

        # Calculate the analogy vector
        result_vector = vec_c + vec_b - vec_a

        # Find top N closest words
        similarities = []
        result_norm = np.linalg.norm(result_vector)

        # Exclude the input words from results
        exclude = [word_a, word_b, word_c]

        for word in self.word_to_idx:
            if word in exclude:
                continue

            word_vec = self.get_vector(word)
            word_norm = np.linalg.norm(word_vec)

            # Cosine similarity
            cosine_sim = np.dot(result_vector, word_vec) / (result_norm * word_norm)

            # Angle in degrees
            angle_radians = np.arccos(np.clip(cosine_sim, -1, 1))
            angle_degrees = np.degrees(angle_radians)

            similarities.append({
                'word': word,
                'cosine_similarity': cosine_sim,
                'angle_degrees': angle_degrees
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)

        return similarities[:top_n], result_vector


def load_and_preprocess_text(filepath):
    """Load HTML file and extract text content"""
    print(f"Loading text from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')

    text_content = soup.get_text()
    text_content = text_content.lower()
    text_content = re.sub(r'\s+', ' ', text_content)

    return text_content


def tokenize_text(text):
    """Tokenize text into sentences and words"""
    print("Tokenizing text...")

    sentences = re.split(r'[.!?]+', text)

    tokenized_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b[a-z]+\b', sentence.lower())
        if len(words) > 1:
            tokenized_sentences.append(words)

    print(f"Total sentences: {len(tokenized_sentences)}")

    total_tokens = sum(len(sent) for sent in tokenized_sentences)
    print(f"Total tokens: {total_tokens}")

    return tokenized_sentences


def generate_analogy_report(word_a, word_b, word_c, results, result_vector, model):
    """Generate a detailed report about the analogy solution"""
    report = []

    report.append("="*80)
    report.append("WORD2VEC ANALOGY SOLVER REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Analogy Problem
    report.append("ANALOGY PROBLEM")
    report.append("-"*40)
    report.append(f'"{word_a}" is to "{word_b}" as "{word_c}" is to ???')
    report.append("")
    report.append("Vector arithmetic formula:")
    report.append(f"result = {word_c} + {word_b} - {word_a}")
    report.append("")

    # Vector Analysis
    report.append("VECTOR ANALYSIS")
    report.append("-"*40)

    vec_a = model.get_vector(word_a)
    vec_b = model.get_vector(word_b)
    vec_c = model.get_vector(word_c)

    if vec_a is not None:
        report.append(f'"{word_a}" vector norm: {np.linalg.norm(vec_a):.6f}')
    if vec_b is not None:
        report.append(f'"{word_b}" vector norm: {np.linalg.norm(vec_b):.6f}')
    if vec_c is not None:
        report.append(f'"{word_c}" vector norm: {np.linalg.norm(vec_c):.6f}')

    report.append(f"Result vector norm: {np.linalg.norm(result_vector):.6f}")
    report.append("")

    # Relationship Analysis
    report.append("RELATIONSHIP ANALYSIS")
    report.append("-"*40)

    if vec_a is not None and vec_b is not None:
        # Analyze king -> queen transformation
        diff_ab = vec_b - vec_a
        report.append(f'"{word_a}" → "{word_b}" transformation:')
        report.append(f"  Vector difference norm: {np.linalg.norm(diff_ab):.6f}")

        # Cosine similarity between king and queen
        cos_sim_ab = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        angle_ab = np.degrees(np.arccos(np.clip(cos_sim_ab, -1, 1)))
        report.append(f"  Cosine similarity: {cos_sim_ab:.6f}")
        report.append(f"  Angle between vectors: {angle_ab:.2f}°")

    report.append("")

    # Solution
    report.append("="*80)
    report.append("ANALOGY SOLUTION")
    report.append("="*80)
    report.append("")

    if results and len(results) > 0:
        answer = results[0]['word']
        report.append(f'ANSWER: "{word_a}" is to "{word_b}" as "{word_c}" is to "{answer.upper()}"')
        report.append("")

        report.append("TOP 5 CANDIDATE WORDS")
        report.append("-"*40)
        report.append("(Ranked by cosine similarity to result vector)")
        report.append("")

        for i, result in enumerate(results, 1):
            report.append(f"{i}. {result['word']}")
            report.append(f"   Cosine Similarity: {result['cosine_similarity']:.6f}")
            report.append(f"   Angle from result vector: {result['angle_degrees']:.2f}°")
            report.append("")

    # Interpretation
    report.append("INTERPRETATION AND ANALYSIS")
    report.append("-"*40)

    if results and len(results) > 0:
        top_word = results[0]['word']
        top_sim = results[0]['cosine_similarity']

        report.append(f'The word "{top_word}" best completes the analogy with a cosine')
        report.append(f"similarity of {top_sim:.6f} to the computed vector.")
        report.append("")

        report.append("SEMANTIC INTERPRETATION:")
        report.append(f'• The relationship "{word_a}" → "{word_b}" represents a')

        # Analyze the semantic relationship
        if word_a == "king" and word_b == "queen":
            report.append("  gender transformation from male royalty to female royalty.")
            report.append("")
            report.append(f'• Applying this same transformation to "{word_c}" yields "{top_word}",')
            report.append("  suggesting a similar gender-based relationship.")

        report.append("")

        # Check if the answer makes semantic sense
        if word_c == "boy" and top_word == "girl":
            report.append("✓ The analogy is semantically correct:")
            report.append('  "boy" (male child) → "girl" (female child)')
            report.append('  parallels "king" (male ruler) → "queen" (female ruler)')
        elif word_c == "man" and top_word == "woman":
            report.append("✓ The analogy is semantically correct:")
            report.append('  "man" (male adult) → "woman" (female adult)')
            report.append('  parallels "king" (male ruler) → "queen" (female ruler)')
        else:
            report.append(f"The model found '{top_word}' as the best match, which may")
            report.append("reflect the specific patterns in Shakespeare's text corpus.")

    report.append("")

    # Vector Geometry Explanation
    report.append("VECTOR GEOMETRY EXPLANATION")
    report.append("-"*40)
    report.append("The analogy works through vector arithmetic in the embedding space:")
    report.append("")
    report.append("1. The difference vector (queen - king) captures the concept of")
    report.append("   'female counterpart' or 'gender transformation'.")
    report.append("")
    report.append("2. Adding this difference to 'boy' moves it in the same direction")
    report.append("   in the semantic space, ideally landing near 'girl'.")
    report.append("")
    report.append("3. This demonstrates that Word2Vec captures not just word meanings")
    report.append("   but also semantic relationships between words.")

    report.append("")
    report.append("METHODOLOGY")
    report.append("-"*40)
    report.append("• Model: Skip-gram with negative sampling")
    report.append("• Training corpus: Shakespeare's Complete Works")
    report.append("• Vector dimensions: 100")
    report.append("• Similarity metric: Cosine similarity")
    report.append(f"• Vocabulary size: {len(model.word_to_idx)} words")

    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    return "\n".join(report)


def main():
    """Main execution function"""
    # File path
    filepath = '../The Complete Works of William Shakespeare.html'

    # Load and preprocess
    text_content = load_and_preprocess_text(filepath)

    # Tokenize
    tokenized_sentences = tokenize_text(text_content)

    # Limit for faster training - but use more to get better coverage
    print("\nUsing first 20000 sentences for training (increased for better vocabulary)...")
    tokenized_sentences = tokenized_sentences[:20000]

    # Train model
    print("\nTraining Word2Vec model...")
    model = Word2Vec(
        vector_size=100,
        window=5,
        min_count=5,
        negative_samples=5,
        learning_rate=0.025,
        epochs=3
    )

    model.train(tokenized_sentences)

    # Solve the analogy
    print("\n" + "="*60)
    print("SOLVING ANALOGY")
    print("="*60)

    word_a = "king"
    word_b = "queen"
    word_c = "boy"

    print(f'Problem: "{word_a}" is to "{word_b}" as "{word_c}" is to ???')
    print(f"Vector arithmetic: {word_c} + {word_b} - {word_a}")

    # Check if all words exist in vocabulary
    missing_words = []
    for word in [word_a, word_b, word_c]:
        if word not in model.word_to_idx:
            missing_words.append(word)

    if missing_words:
        print(f"\nError: The following words are not in vocabulary: {missing_words}")
        print("Cannot solve analogy.")
        return

    # Solve the analogy
    results, result_vector = model.solve_analogy(word_a, word_b, word_c, top_n=5)

    if results:
        print("\n" + "="*60)
        print("SOLUTION")
        print("="*60)
        print(f'\n"{word_a}" is to "{word_b}" as "{word_c}" is to "{results[0]["word"].upper()}"')

        print("\nTop 5 candidates:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['word']:15s} (similarity: {result['cosine_similarity']:.6f})")

    # Generate report
    print("\n" + "="*60)
    print("Generating analogy report...")
    print("="*60)

    report = generate_analogy_report(word_a, word_b, word_c, results, result_vector, model)

    # Save report
    report_filename = 'analogy_report.txt'
    with open(report_filename, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_filename}")

    # Save data as JSON
    analogy_data = {
        'timestamp': datetime.now().isoformat(),
        'analogy': {
            'word_a': word_a,
            'word_b': word_b,
            'word_c': word_c,
            'solution': results[0]['word'] if results else None
        },
        'top_candidates': [
            {
                'word': r['word'],
                'cosine_similarity': float(r['cosine_similarity']),
                'angle_degrees': float(r['angle_degrees'])
            }
            for r in results
        ] if results else []
    }

    json_filename = 'analogy_data.json'
    with open(json_filename, 'w') as f:
        json.dump(analogy_data, f, indent=2)

    print(f"Analogy data saved to: {json_filename}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()