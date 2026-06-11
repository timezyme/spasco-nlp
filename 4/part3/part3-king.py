#!/usr/bin/env python3
"""
Word2Vec Similarity Analysis for "king"
Finds the 5 most similar words to "king" using cosine similarity
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

    def find_most_similar(self, word, top_n=5):
        """Find most similar words using cosine similarity"""
        if word not in self.word_to_idx:
            return []

        word_vec = self.get_vector(word)
        word_norm = np.linalg.norm(word_vec)

        similarities = []

        # Calculate cosine similarity with all other words
        for other_word in self.word_to_idx:
            if other_word == word:  # Skip the word itself
                continue

            other_vec = self.get_vector(other_word)
            other_norm = np.linalg.norm(other_vec)

            # Cosine similarity = dot product / (norm1 * norm2)
            cosine_sim = np.dot(word_vec, other_vec) / (word_norm * other_norm)

            # Calculate angle in degrees for better interpretation
            # cos(theta) = cosine_sim, so theta = arccos(cosine_sim)
            # Clamp to valid range [-1, 1] to avoid numerical errors
            angle_radians = np.arccos(np.clip(cosine_sim, -1, 1))
            angle_degrees = np.degrees(angle_radians)

            similarities.append({
                'word': other_word,
                'cosine_similarity': cosine_sim,
                'angle_degrees': angle_degrees,
                'vector': other_vec
            })

        # Sort by cosine similarity (descending) - most similar first
        similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)

        return similarities[:top_n]


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


def generate_similarity_report(target_word, similar_words, model):
    """Generate a detailed report about word similarities"""
    report = []

    report.append("="*80)
    report.append("WORD2VEC SIMILARITY ANALYSIS REPORT")
    report.append(f"Finding Words Most Similar to '{target_word.upper()}'")
    report.append("="*80)
    report.append("")
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Target word analysis
    report.append("TARGET WORD ANALYSIS")
    report.append("-"*40)
    target_vec = model.get_vector(target_word)
    if target_vec is not None:
        report.append(f"Word: '{target_word}'")
        report.append(f"Vector norm: {np.linalg.norm(target_vec):.6f}")
        report.append(f"Vector dimensionality: {len(target_vec)}")
        report.append(f"Vocabulary size: {len(model.word_to_idx)} words")
    report.append("")

    # Most similar words
    report.append("TOP 5 MOST SIMILAR WORDS")
    report.append("-"*40)
    report.append("")
    report.append("Ranking by cosine similarity (most collinear vectors):")
    report.append("")

    for i, sim_data in enumerate(similar_words, 1):
        report.append(f"{i}. Word: '{sim_data['word']}'")
        report.append(f"   Cosine Similarity: {sim_data['cosine_similarity']:.6f}")
        report.append(f"   Angle (degrees): {sim_data['angle_degrees']:.2f}°")
        report.append(f"   Interpretation: {'Very similar' if sim_data['cosine_similarity'] > 0.8 else 'Similar' if sim_data['cosine_similarity'] > 0.6 else 'Somewhat similar'}")
        report.append("")

    # Analysis and interpretation
    report.append("="*80)
    report.append("ANALYSIS AND INTERPRETATION")
    report.append("="*80)
    report.append("")

    report.append("COSINE SIMILARITY INTERPRETATION:")
    report.append("-"*40)
    report.append("• Cosine similarity ranges from -1 to 1")
    report.append("• 1.0 = identical vectors (0° angle)")
    report.append("• 0.0 = orthogonal vectors (90° angle)")
    report.append("• -1.0 = opposite vectors (180° angle)")
    report.append("")

    report.append("SEMANTIC RELATIONSHIPS DISCOVERED:")
    report.append("-"*40)

    # Analyze the top similar words
    if similar_words:
        avg_similarity = sum(w['cosine_similarity'] for w in similar_words) / len(similar_words)
        report.append(f"• Average similarity score: {avg_similarity:.6f}")
        report.append(f"• Similarity range: {similar_words[-1]['cosine_similarity']:.6f} to {similar_words[0]['cosine_similarity']:.6f}")
        report.append("")

        # Categorize the similar words
        royal_words = [w['word'] for w in similar_words if w['word'] in ['queen', 'prince', 'lord', 'duke', 'monarch', 'throne', 'crown']]
        if royal_words:
            report.append(f"• Royal/nobility terms found: {', '.join(royal_words)}")

        power_words = [w['word'] for w in similar_words if w['word'] in ['power', 'rule', 'reign', 'authority', 'command']]
        if power_words:
            report.append(f"• Power/authority terms found: {', '.join(power_words)}")

        report.append("")
        report.append("OBSERVATIONS:")
        report.append("-"*40)
        report.append(f"• The word '{target_word}' shows strong semantic relationships with:")

        for w in similar_words[:3]:  # Focus on top 3
            report.append(f"  - '{w['word']}' (angle: {w['angle_degrees']:.1f}°)")

        report.append("")
        report.append("• These relationships suggest that the Word2Vec model has successfully")
        report.append("  captured semantic and contextual patterns from Shakespeare's text.")

        if similar_words[0]['cosine_similarity'] > 0.7:
            report.append("")
            report.append("• The high similarity scores (>0.7) indicate that these words")
            report.append("  frequently appear in similar contexts throughout the corpus.")

    report.append("")
    report.append("METHODOLOGY NOTES:")
    report.append("-"*40)
    report.append("• Model: Skip-gram with negative sampling")
    report.append("• Training corpus: Shakespeare's Complete Works")
    report.append("• Context window: 5 words")
    report.append("• Vector dimensions: 100")
    report.append("• Similarity metric: Cosine similarity (dot product of normalized vectors)")
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

    # Limit for faster training
    print("\nUsing first 10000 sentences for training...")
    tokenized_sentences = tokenized_sentences[:10000]

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

    # Find most similar words to "king"
    target_word = "king"
    print(f"\n\nFinding words most similar to '{target_word}'...")

    similar_words = model.find_most_similar(target_word, top_n=5)

    # Display results
    print("\n" + "="*60)
    print(f"Top 5 words most similar to '{target_word}':")
    print("="*60)

    for i, sim_data in enumerate(similar_words, 1):
        print(f"\n{i}. {sim_data['word']}")
        print(f"   Cosine Similarity: {sim_data['cosine_similarity']:.6f}")
        print(f"   Angle: {sim_data['angle_degrees']:.2f} degrees")

    # Generate and save report
    print("\n" + "="*60)
    print("Generating similarity report...")
    print("="*60)

    report = generate_similarity_report(target_word, similar_words, model)

    # Save report
    report_filename = 'similarity_report_king.txt'
    with open(report_filename, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_filename}")

    # Save similarity data as JSON
    similarity_data = {
        'target_word': target_word,
        'timestamp': datetime.now().isoformat(),
        'similar_words': [
            {
                'word': w['word'],
                'cosine_similarity': float(w['cosine_similarity']),
                'angle_degrees': float(w['angle_degrees'])
            }
            for w in similar_words
        ]
    }

    json_filename = 'similarity_data_king.json'
    with open(json_filename, 'w') as f:
        json.dump(similarity_data, f, indent=2)

    print(f"Similarity data saved to: {json_filename}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()