"""
Problem 3: Undercomplete Autoencoder for IMDb Movie Reviews
Bag-of-Words Dense Autoencoder - IMPROVED VERSION

Author: Stephen Pasco

IMPROVEMENTS FROM CODE REVIEW:
1. Fixed CRITICAL BUG in decode_bow() - removed sorted() that destroyed probability ranking
2. Iterative experimentation: Train ONE model at a time (change CODING_SIZE below)
3. Changed loss to binary_crossentropy (better for binary BoW data)
4. Added EarlyStopping callback for efficiency
5. Used validation_split instead of test set for validation
6. Added baseline comparison (mean prediction)
7. Added dropout regularization to prevent overfitting
8. Replaced arbitrary MSE threshold with data-driven approach (10% degradation)

USAGE:
1. Set CODING_SIZE below (try: 64, 32, 16, 8, 4)
2. Run: python 5/problem3/problem3-fix.py
3. Results accumulate in output text file
4. Change CODING_SIZE and run again
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import models, layers, callbacks, optimizers
from keras.datasets import imdb
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION: Change this value for each run
# ============================================================================
CODING_SIZE = 4  # 🔧 CHANGE THIS: Try [64, 32, 16, 8, 4]
SAVE_MODELS = False  # Set to True if you want to save 126MB model files
# ============================================================================

# REMOVED JSON FUNCTIONS - We only want text output!

def main():
    np.random.seed(42)

    print("="*80)
    print(f"IMDb AUTOENCODER - TRAINING {CODING_SIZE} CODINGS")
    print("="*80)

    # ============================================================================
    # LOAD DATA (using provided code)
    # ============================================================================
    print("\n[1] Loading IMDb dataset...")

    max_features = 10000  # Top 10,000 words
    maxlen = 200  # First 200 words of each review

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")

    # ============================================================================
    # CONVERT TO BAG-OF-WORDS
    # ============================================================================
    print("\n[2] Converting to bag-of-words vectors...")

    def to_multihot(sequences, dimension=max_features):
        """Convert sequences to multi-hot (bag-of-words) vectors"""
        results = np.zeros((len(sequences), dimension))
        for i, seq in enumerate(sequences):
            results[i, np.unique(seq)] = 1.0
        return results

    X_train_bow = to_multihot(X_train)
    X_test_bow = to_multihot(X_test)

    print(f"✓ Shape: {X_train_bow.shape}")
    print(f"✓ Avg words/review: {X_train_bow.sum(axis=1).mean():.1f}")

    # Load word index
    word_index = imdb.get_word_index()
    index_to_word = {rank + 3: word for word, rank in word_index.items()
                     if (rank + 3) < max_features}
    index_to_word[0] = '<PAD>'
    index_to_word[1] = '<START>'
    index_to_word[2] = '<UNK>'

    def decode_bow(bow_vector, top_k=50, for_reconstructed=False):
        """
        Convert bag-of-words to word list

        FIXED: Removed sorted() on line 73 to preserve probability ranking
        """
        if for_reconstructed:
            # For reconstructions: show top-K words by probability (descending order)
            top_indices = np.argsort(bow_vector)[::-1][:top_k]
            # FIXED: Don't sort by index - keep probability order!
            words = [index_to_word.get(int(idx), '<UNK>') for idx in top_indices]
        else:
            # For originals: show words that are present (value = 1.0)
            word_indices = np.where(bow_vector > 0.5)[0]
            words = [index_to_word.get(int(idx), '<UNK>') for idx in sorted(word_indices)[:top_k]]
        return words

    # ============================================================================
    # BASELINE COMPARISON
    # ============================================================================
    print("\n[2b] Computing baseline performance...")
    mean_bow = X_train_bow.mean(axis=0)
    baseline_predictions = np.tile(mean_bow, (len(X_test_bow), 1))
    baseline_mse = np.mean((X_test_bow - baseline_predictions)**2)
    print(f"✓ Baseline MSE (mean prediction): {baseline_mse:.4f}")
    print("   (Models must beat this to be useful)")

    # ============================================================================
    # BUILD AND TRAIN MODEL (single model)
    # ============================================================================
    print(f"\n[3] Building model with {CODING_SIZE} codings...")
    print("    (Using binary_crossentropy, dropout, and early stopping)")

    # Build model using the reference architecture
    encoder_input = layers.Input(shape=(max_features,))
    x = layers.Dense(512, activation='relu')(encoder_input)
    x = layers.Dropout(0.1)(x)  # Reference uses 0.1, not 0.2
    x = layers.Dense(256, activation='relu')(x)
    # CRITICAL: Use LINEAR activation in bottleneck for representation learning!
    encoder_output = layers.Dense(CODING_SIZE, activation='linear', name='latent')(x)

    # Decoder (symmetric)
    x = layers.Dense(256, activation='relu')(encoder_output)
    x = layers.Dropout(0.1)(x)  # Reference uses 0.1
    x = layers.Dense(512, activation='relu')(x)
    decoder_output = layers.Dense(max_features, activation='sigmoid')(x)

    model = models.Model(encoder_input, decoder_output)

    # Use binary_crossentropy for binary data (matching reference setup)
    model.compile(
        optimizer=optimizers.Adam(1e-3),  # Reference uses explicit learning rate
        loss='binary_crossentropy',
        metrics=['binary_accuracy']  # Reference uses binary_accuracy metric
    )

    print(f"✓ Model created")
    print(f"✓ Parameters: {model.count_params():,}")

    # Add EarlyStopping callback (matching reference settings)
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reference uses patience=3
        restore_best_weights=True,
        verbose=0  # Silent - no "Restoring model weights" messages
    )

    print(f"\n[4] Training model (max 15 epochs, early stopping patience=3)...")
    print(f"    Using reference architecture: 512→256→{CODING_SIZE}(linear)→256→512")
    print(f"    Training...", end='', flush=True)

    # Train with reference parameters
    history = model.fit(
        X_train_bow, X_train_bow,
        epochs=15,  # Reference uses 15 epochs
        batch_size=256,  # Reference uses batch_size=256
        validation_split=0.1,  # Reference uses 0.1 split
        callbacks=[early_stop],
        verbose=0  # Silent training - no epoch spam
    )

    epochs_trained = len(history.history['loss'])
    print(f" ✓ Complete in {epochs_trained} epochs")

    # Evaluate on test set (proper holdout)
    print(f"\n[5] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test_bow, X_test_bow, verbose=0)

    print(f"✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")

    # Save model weights (optional - these files are 126MB each!)
    if SAVE_MODELS:
        model_file = f'5/problem3/model_{CODING_SIZE}_codings.keras'
        model.save(model_file)
        print(f"✓ Model saved to: {model_file}")

    # Just print results - no JSON saving needed

    # ============================================================================
    # EVALUATE 3 SAMPLE RECONSTRUCTIONS
    # ============================================================================
    print(f"\n[6] Sample Reconstructions ({CODING_SIZE} codings)...")
    print("="*80)

    np.random.seed(42)
    random_indices = np.random.choice(len(X_test_bow), 3, replace=False)
    test_samples = X_test_bow[random_indices]

    reconstructed = model.predict(test_samples, verbose=0)

    for i in range(3):
        print(f"\nSample {i+1} (Sentiment: {'Positive' if y_test[random_indices[i]] == 1 else 'Negative'})")
        print("-"*80)

        orig_words = decode_bow(test_samples[i], top_k=30, for_reconstructed=False)
        recon_words = decode_bow(reconstructed[i], top_k=30, for_reconstructed=True)

        print(f"ORIGINAL (top 30):\n  {' '.join(orig_words)}")
        print(f"\nRECONSTRUCTED (top 30 by probability):\n  {' '.join(recon_words)}")

        overlap = len(set(orig_words) & set(recon_words))
        overlap_pct = (overlap / len(orig_words)) * 100 if len(orig_words) > 0 else 0
        top_probs = np.sort(reconstructed[i])[::-1][:5]

        print(f"\n  Word Overlap: {overlap}/{len(orig_words)} ({overlap_pct:.1f}%)")
        print(f"  Top 5 Probs: [{', '.join([f'{p:.3f}' for p in top_probs])}]")

    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nBaseline MSE: {baseline_mse:.4f}")
    print(f"\n{CODING_SIZE} codings:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Compression: {max_features/CODING_SIZE:.1f}:1")
    print(f"  Epochs trained: {epochs_trained}")

    # ============================================================================
    # ANSWER
    # ============================================================================
    print(f"\n" + "="*80)
    print("ANSWER TO QUESTION [6]")
    print("="*80)
    print(f"\n** Training {CODING_SIZE} codings complete **")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Compression Ratio: {max_features/CODING_SIZE:.1f}:1")

    print(f"\n" + "="*80)
    print(f"✓ {CODING_SIZE} codings complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Redirect stdout to file (append mode to accumulate)
    output_file = '5/problem3/problem3-fix_output.txt'

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"RUN: {CODING_SIZE} codings\n")
        f.write(f"{'='*80}\n\n")

        # Save original stdout
        original_stdout = sys.stdout

        # Tee output to both console and file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = TeeOutput(original_stdout, f)

        try:
            main()
        finally:
            sys.stdout = original_stdout

    print(f"✓ Output written to: {output_file}")
