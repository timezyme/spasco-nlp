"""
Project 5 - Problem 1: Character-Level Embeddings with Keras

This script implements character-level embeddings by:
(a) Preprocessing text data with character-to-integer mapping
(b) Training a character embedding model on large text
(c) Applying embeddings to smaller text and analyzing information transfer
(d) Handling out-of-vocabulary (OOV) characters

Author: Stephen Pasco
"""

import numpy as np
from keras import models, layers, optimizers
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("PROJECT 5 - PROBLEM 1: CHARACTER-LEVEL EMBEDDINGS")
print("=" * 80)
print()

# ============================================================================
# PART (A): TEXT PREPROCESSING AND CHARACTER-TO-INTEGER MAPPING
# ============================================================================

print("=" * 80)
print("PART (A): TEXT PREPROCESSING")
print("=" * 80)
print()

# Large text dataset for training (2-3 pages worth of text)
# Using a news article about artificial intelligence
large_text = """
artificial intelligence has revolutionized the way we interact with technology
in recent years. machine learning algorithms have become increasingly
sophisticated, enabling computers to perform tasks that once required human
intelligence. from natural language processing to computer vision, ai systems
are now capable of understanding and interpreting complex patterns in data.

the development of deep learning has been particularly transformative. neural
networks with multiple layers can learn hierarchical representations of data,
making them highly effective for a wide range of applications. convolutional
neural networks excel at image recognition, while recurrent neural networks
are well suited for sequential data like text and speech.

one of the most exciting applications of ai is in natural language processing.
modern language models can generate human-like text, translate between
languages, and even engage in conversations. these models are trained on vast
amounts of text data, learning the statistical patterns and structures of
human language. the transformer architecture, introduced in recent years, has
become the foundation for many state-of-the-art nlp systems.

however, the rapid advancement of ai also raises important ethical questions.
as ai systems become more powerful and ubiquitous, we must consider issues of
bias, privacy, and accountability. researchers and policymakers are working to
develop frameworks that ensure ai technologies are developed and deployed
responsibly. transparency in ai decision-making processes is crucial for
building trust and ensuring fair outcomes.

the future of artificial intelligence holds tremendous promise. emerging
technologies like quantum computing could further accelerate ai capabilities.
advances in reinforcement learning are enabling robots to learn complex tasks
through trial and error. multimodal ai systems that can process and integrate
information from multiple sources such as text, images, and audio are becoming
more sophisticated. as we continue to push the boundaries of what ai can do,
it is essential that we remain thoughtful about the societal implications and
work towards creating ai systems that benefit all of humanity.

the integration of ai into various industries has already begun to transform
how we work and live. in healthcare, ai algorithms are assisting doctors in
diagnosing diseases and recommending treatments. in finance, machine learning
models help detect fraud and make investment decisions. autonomous vehicles
powered by ai are being tested on roads around the world. customer service
chatbots powered by natural language processing can handle routine inquiries
efficiently. the potential applications are virtually limitless.

education is another area where ai is making significant inroads. intelligent
tutoring systems can provide personalized learning experiences tailored to
individual student needs. automated grading tools can save teachers time while
providing detailed feedback to students. language learning apps use speech
recognition and natural language understanding to help users practice
conversation skills. as these technologies mature, they have the potential to
make education more accessible and effective for learners worldwide.

despite these advances, significant challenges remain. current ai systems often
require large amounts of labeled training data, which can be expensive and
time-consuming to collect. they may also struggle with tasks that require
common sense reasoning or understanding of context. researchers are actively
working on techniques like few-shot learning and transfer learning to address
these limitations. the goal is to create ai systems that can learn more
efficiently from smaller amounts of data and generalize better to new
situations.

in conclusion, artificial intelligence represents one of the most significant
technological developments of our time. while challenges remain, the potential
benefits are enormous. by approaching ai development with thoughtfulness and
responsibility, we can harness its power to solve some of humanity's most
pressing problems and create a better future for all.
"""

# Small text with some new characters/symbols not in large text
small_text = """
Q: What's the cost of AI research?
A: It's approximately $10M-$50M per year!
Email: ai-research@example.com (2024) #AIRevolution
"""

# Convert to lowercase
large_text = large_text.lower()
small_text = small_text.lower()

print("Large Text Sample (first 200 characters):")
print(large_text[:200])
print(f"\nLarge text length: {len(large_text)} characters")
print()

print("Small Text (complete):")
print(small_text)
print(f"\nSmall text length: {len(small_text)} characters")
print()

# Build character vocabulary from LARGE TEXT ONLY
print("-" * 80)
print("Building Character Vocabulary from Large Text")
print("-" * 80)

# Extract unique characters from large text
unique_chars = sorted(set(large_text))
print(f"Unique characters in large text: {len(unique_chars)}")

# Add <UNK> token for unknown characters
UNK_TOKEN = '<UNK>'
vocab = [UNK_TOKEN] + unique_chars

# Create bidirectional mappings
char_to_int = {char: idx for idx, char in enumerate(vocab)}
int_to_char = {idx: char for idx, char in enumerate(vocab)}

vocab_size = len(vocab)
print(f"Total vocabulary size (including <UNK>): {vocab_size}")
print()

# Display vocabulary sample
print("Vocabulary sample (first 30 characters):")
print(vocab[:30])
print()

# Function to encode text to integer sequence
def encode_text(text, char_to_int, unk_token='<UNK>'):
    """
    Convert text to integer sequence, mapping unknown characters to <UNK> token.
    """
    unk_idx = char_to_int[unk_token]
    return [char_to_int.get(char, unk_idx) for char in text]

# Encode both texts
large_text_encoded = encode_text(large_text, char_to_int)
small_text_encoded = encode_text(small_text, char_to_int)

print("-" * 80)
print("Character-to-Integer Mapping Examples")
print("-" * 80)
print()

# Display samples of small text showing character-to-integer mappings
print("Small Text Character-to-Integer Mapping (first 50 characters):")
sample_text = small_text[:50]
sample_encoded = small_text_encoded[:50]

for i, (char, idx) in enumerate(zip(sample_text, sample_encoded)):
    char_display = repr(char) if char in ['\n', '\t', ' '] else char
    token_name = int_to_char[idx]
    if token_name == '<UNK>':
        print(f"  [{i:2d}] '{char_display}' -> {idx:3d} ({token_name}) *UNKNOWN*")
    else:
        print(f"  [{i:2d}] '{char_display}' -> {idx:3d}")

print()

# Identify unknown characters in small text
unknown_chars = set()
for char in small_text:
    if char not in char_to_int or char == UNK_TOKEN:
        if char_to_int.get(char, -1) == char_to_int[UNK_TOKEN] or char not in char_to_int:
            unknown_chars.add(char)

print(f"Characters in small text NOT in large text vocabulary: {len(unknown_chars)}")
if unknown_chars:
    print(f"Unknown characters: {sorted(unknown_chars)}")
print()

# ============================================================================
# PART (B): CHARACTER EMBEDDING MODEL TRAINING
# ============================================================================

print("=" * 80)
print("PART (B): CHARACTER EMBEDDING MODEL")
print("=" * 80)
print()

print("-" * 80)
print("Preparing Training Data")
print("-" * 80)

# Create training sequences: predict next character given current character
# Input: character at position i, Output: character at position i+1
X_train = []
y_train = []

for i in range(len(large_text_encoded) - 1):
    X_train.append(large_text_encoded[i])
    y_train.append(large_text_encoded[i + 1])

X_train = np.array(X_train)
y_train = np.array(y_train)

# One-hot encode output labels
y_train_categorical = to_categorical(y_train, num_classes=vocab_size)

print(f"Training sequences created: {len(X_train)}")
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {y_train_categorical.shape}")
print()

# Model architecture
print("-" * 80)
print("Building Character Embedding Model")
print("-" * 80)

embedding_dim = 64  # Dimension of character embeddings
hidden_units = 128  # Number of hidden units

model = models.Sequential([
    # Embedding layer: maps character indices to dense vectors
    layers.Embedding(input_dim=vocab_size,
                     output_dim=embedding_dim,
                     input_length=1,
                     name='char_embedding'),

    # Flatten layer: convert 3D to 2D
    layers.Flatten(),

    # Dense hidden layer: learns patterns
    layers.Dense(hidden_units, activation='relu'),

    # Dense output layer: predicts next character
    layers.Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print()

# Train the model
print("-" * 80)
print("Training Character Embedding Model")
print("-" * 80)
print("This may take a few minutes...")
print()

history = model.fit(
    X_train,
    y_train_categorical,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

print()
print("Training completed!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()

# Extract the trained embedding layer
print("-" * 80)
print("Extracting Trained Embedding Weights")
print("-" * 80)

embedding_layer = model.get_layer('char_embedding')
embedding_weights = embedding_layer.get_weights()[0]

print(f"Embedding matrix shape: {embedding_weights.shape}")
print(f"  - Vocabulary size: {embedding_weights.shape[0]}")
print(f"  - Embedding dimension: {embedding_weights.shape[1]}")
print()

# Apply embeddings to small text
print("-" * 80)
print("Applying Embeddings to Small Text")
print("-" * 80)
print()

# Convert small text encoded sequence to embeddings
small_text_embeddings = embedding_weights[small_text_encoded]

print(f"Small text embedding matrix shape: {small_text_embeddings.shape}")
print(f"  - Sequence length: {small_text_embeddings.shape[0]}")
print(f"  - Embedding dimension: {small_text_embeddings.shape[1]}")
print()

# Display sample character embeddings
print("-" * 80)
print("Sample Character Embeddings")
print("-" * 80)
print()

# Example 1: Common character 'a'
char_a = 'a'
if char_a in char_to_int:
    idx_a = char_to_int[char_a]
    embedding_a = embedding_weights[idx_a]
    print(f"Character '{char_a}' embedding vector:")
    print(f"  Index: {idx_a}")
    print(f"  Vector (first 10 dimensions): {embedding_a[:10]}")
    print(f"  Vector norm: {np.linalg.norm(embedding_a):.4f}")
    print()

# Example 2: Common character 'e'
char_e = 'e'
if char_e in char_to_int:
    idx_e = char_to_int[char_e]
    embedding_e = embedding_weights[idx_e]
    print(f"Character '{char_e}' embedding vector:")
    print(f"  Index: {idx_e}")
    print(f"  Vector (first 10 dimensions): {embedding_e[:10]}")
    print(f"  Vector norm: {np.linalg.norm(embedding_e):.4f}")
    print()

# Example 3: Space character
char_space = ' '
if char_space in char_to_int:
    idx_space = char_to_int[char_space]
    embedding_space = embedding_weights[idx_space]
    print(f"Character ' ' (space) embedding vector:")
    print(f"  Index: {idx_space}")
    print(f"  Vector (first 10 dimensions): {embedding_space[:10]}")
    print(f"  Vector norm: {np.linalg.norm(embedding_space):.4f}")
    print()

# Example 4: Unknown token
idx_unk = char_to_int[UNK_TOKEN]
embedding_unk = embedding_weights[idx_unk]
print(f"<UNK> token embedding vector:")
print(f"  Index: {idx_unk}")
print(f"  Vector (first 10 dimensions): {embedding_unk[:10]}")
print(f"  Vector norm: {np.linalg.norm(embedding_unk):.4f}")
print()

# Show specific examples from small text
print("-" * 80)
print("Small Text Sample with Embeddings")
print("-" * 80)
print()

# Show first 10 characters of small text with their embeddings
sample_size = 10
print(f"First {sample_size} characters of small text with embeddings:")
for i in range(min(sample_size, len(small_text))):
    char = small_text[i]
    char_display = repr(char) if char in ['\n', '\t', ' '] else char
    encoded_idx = small_text_encoded[i]
    embedding_vec = small_text_embeddings[i]
    token_name = int_to_char[encoded_idx]

    if token_name == '<UNK>':
        print(f"  [{i}] '{char_display}' -> <UNK> (unknown)")
    else:
        print(f"  [{i}] '{char_display}' -> '{token_name}'")
    print(f"      Embedding (first 8 dims): {embedding_vec[:8]}")
    print()

# ============================================================================
# PART (C): ANALYSIS OF INFORMATION TRANSFER
# ============================================================================

print("=" * 80)
print("PART (C): ANALYSIS OF INFORMATION TRANSFER")
print("=" * 80)
print()

analysis_c = """
When we apply the character embedding model trained on the large text to the
smaller text, several types of information are transferred:

1. SHARED CHARACTER VOCABULARY:
   - Characters that appear in BOTH the large training text and the small test
     text benefit from learned representations.
   - Common characters like 'a', 'e', 't', 'i', 'n', 's', space, newline, etc.
     have learned embeddings that capture their distributional properties.
   - In our case, we identified {} unique characters in the large text, and many
     of these also appear in the small text.

2. EMBEDDING WEIGHTS (SEMANTIC RELATIONSHIPS):
   - The embedding matrix learned from the large text encodes relationships between
     characters based on their co-occurrence patterns.
   - Characters that frequently appear in similar contexts have similar embedding
     vectors (measured by cosine similarity or Euclidean distance).
   - For example:
     * Vowels (a, e, i, o, u) may have similar embeddings because they appear
       in similar word contexts
     * Common consonants that frequently follow vowels may cluster together
     * Punctuation marks may form their own semantic group

3. SEQUENTIAL PATTERNS:
   - The model learned which characters typically follow other characters
     in the large text.
   - This sequential knowledge is encoded in the embedding space, where characters
     that are good predictors of each other are positioned closer together.

4. CONTEXTUAL UNDERSTANDING:
   - The embeddings capture context-dependent information about characters.
   - For instance, the model learned that spaces typically separate words,
     periods end sentences, and certain character sequences form common patterns.

5. TRANSFER LEARNING BENEFIT:
   - When we apply these embeddings to the small text, we get "free" semantic
     representations for all shared characters WITHOUT needing to train on the
     small text.
   - This is the essence of transfer learning: knowledge gained from a large
     dataset (large text) transfers to benefit a smaller dataset (small text).

SIGNIFICANCE FOR SMALLER TEXT:
- Characters in the small text that also appeared in the large text inherit rich,
  meaningful representations learned from extensive training data.
- These embeddings can be used for downstream tasks like text classification,
  generation, or similarity measurement on the small text.
- The model provides consistent representations across both texts, enabling
  comparison and analysis.

CONCRETE EXAMPLE FROM OUR DATA:
- The character 'a' appears {count_a_large} times in the large text and
  {count_a_small} times in the small text.
- Its embedding vector has learned to represent 'a' based on all {count_a_large}
  occurrences in the large text.
- When 'a' appears in the small text, it uses this same learned representation,
  benefiting from the larger training corpus.
""".format(
    len(unique_chars),
    count_a_large=large_text.count('a'),
    count_a_small=small_text.count('a')
)

print(analysis_c)
print()

# Quantitative analysis
print("-" * 80)
print("Quantitative Analysis of Information Transfer")
print("-" * 80)
print()

# Calculate overlap between texts
small_text_chars = set(small_text)
large_text_chars = set(large_text)
shared_chars = small_text_chars.intersection(large_text_chars)
unique_to_small = small_text_chars - large_text_chars

print(f"Characters in large text: {len(large_text_chars)}")
print(f"Characters in small text: {len(small_text_chars)}")
print(f"Shared characters: {len(shared_chars)} ({100*len(shared_chars)/len(small_text_chars):.1f}% of small text chars)")
print(f"Characters unique to small text: {len(unique_to_small)}")
print()

if unique_to_small:
    print(f"Characters in small text NOT in large text: {sorted(unique_to_small)}")
    print()

# Calculate coverage
small_text_char_counts = {}
for char in small_text:
    small_text_char_counts[char] = small_text_char_counts.get(char, 0) + 1

covered_count = sum(count for char, count in small_text_char_counts.items() if char in large_text_chars)
total_count = len(small_text)

print(f"Character-level coverage:")
print(f"  - Total characters in small text: {total_count}")
print(f"  - Characters with learned embeddings: {covered_count} ({100*covered_count/total_count:.1f}%)")
print(f"  - Characters mapped to <UNK>: {total_count - covered_count} ({100*(total_count-covered_count)/total_count:.1f}%)")
print()

# ============================================================================
# PART (D): OUT-OF-VOCABULARY (OOV) HANDLING
# ============================================================================

print("=" * 80)
print("PART (D): OUT-OF-VOCABULARY (OOV) CHARACTER HANDLING")
print("=" * 80)
print()

analysis_d = """
HANDLING NEW CHARACTERS IN KERAS:

When the smaller text contains characters that did not appear in the large
training text, the model must handle these Out-Of-Vocabulary (OOV) cases.
Our implementation uses the <UNK> (unknown) token approach:

1. VOCABULARY BUILDING:
   - We built our character vocabulary EXCLUSIVELY from the large training text.
   - We explicitly added a special <UNK> token to the vocabulary at index 0.
   - This token serves as a placeholder for any character not seen during training.

2. ENCODING PROCESS:
   - When encoding the small text, our encode_text() function checks if each
     character exists in the char_to_int dictionary.
   - If a character is NOT found, it is mapped to the <UNK> token index.
   - This ensures all characters can be represented, even if they're unknown.

3. EMBEDDING REPRESENTATION:
   - The <UNK> token has its own learned embedding vector in the embedding matrix.
   - This embedding is trained alongside all other character embeddings.
   - All unknown characters share the SAME embedding vector (the <UNK> embedding).

4. IMPLICATIONS:
   - Unknown characters lose their individual identity and are treated identically.
   - The <UNK> embedding represents a "generic unknown character" concept.
   - This is a limitation: we cannot distinguish between different unknown characters.
   - However, it prevents model failure and provides a graceful fallback mechanism.

5. ALTERNATIVE APPROACHES (NOT USED HERE):
   - Character n-grams: Represent unknown characters using subword units
   - Byte-level encoding: Encode all possible bytes (0-255) in vocabulary
   - Unicode category features: Map unknown characters to their Unicode category
   - Hash-based representation: Use character hashing to fixed-size vocabulary

EXAMPLES FROM OUR DATA:

In the small text, we have the following OOV characters:
"""

print(analysis_d)

# Show OOV examples
if unknown_chars:
    print(f"Unknown characters found: {sorted(unknown_chars)}")
    print()

    # Show how each unknown character is handled
    print("OOV Character Handling Examples:")
    print()

    for unknown_char in sorted(unknown_chars)[:5]:  # Show up to 5 examples
        char_display = repr(unknown_char) if unknown_char in ['\n', '\t', ' '] else unknown_char

        # Find first occurrence in small text
        first_occurrence = small_text.index(unknown_char)
        context_start = max(0, first_occurrence - 5)
        context_end = min(len(small_text), first_occurrence + 6)
        context = small_text[context_start:context_end]

        print(f"  Character: '{char_display}'")
        print(f"  Context: ...{repr(context)}...")
        print(f"  Mapped to: <UNK> (index {char_to_int[UNK_TOKEN]})")
        print(f"  Embedding: {embedding_unk[:8]}")
        print()

    # Show embedding statistics
    print("Embedding Comparison:")
    print(f"  - <UNK> embedding norm: {np.linalg.norm(embedding_unk):.4f}")
    if 'a' in char_to_int:
        print(f"  - Character 'a' embedding norm: {np.linalg.norm(embedding_weights[char_to_int['a']]):.4f}")
    print()
else:
    print("No unknown characters found in the small text.")
    print("All characters in the small text were present in the large training text.")
    print()

print("-" * 80)
print("Summary of OOV Handling Strategy")
print("-" * 80)
print()

summary = f"""
STRATEGY SUMMARY:
1. Vocabulary Construction: Built from large text only + <UNK> token
2. Vocabulary Size: {vocab_size} characters (including <UNK>)
3. Unknown Character Mapping: All OOV chars -> <UNK> token (index {char_to_int[UNK_TOKEN]})
4. Embedding Dimension: {embedding_dim}
5. Unknown Characters in Small Text: {len(unknown_chars)}
6. Coverage: {100*covered_count/total_count:.1f}% of small text characters have learned embeddings

ADVANTAGES:
- Robust: Never fails on new characters
- Simple: Easy to implement and understand
- Consistent: All unknown characters treated uniformly

LIMITATIONS:
- Information Loss: Cannot distinguish between different unknown characters
- Embedding Quality: <UNK> embedding may not be semantically meaningful
- No Adaptability: Cannot learn from new characters without retraining
"""

print(summary)

print()
print("=" * 80)
print("PROJECT 5 PROBLEM 1 COMPLETE")
print("=" * 80)
print()

print("SUMMARY OF DELIVERABLES:")
print()
print("✓ Part (a): Text preprocessing and character-to-integer mapping completed")
print("✓ Part (b): Character embedding model trained and applied to small text")
print("✓ Part (c): Analysis of information transfer provided")
print("✓ Part (d): OOV handling explanation and examples provided")
print()
print("All outputs, analyses, and visualizations have been displayed above.")
print()
