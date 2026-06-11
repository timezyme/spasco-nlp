"""
Problem 3: Undercomplete Autoencoder for IMDb Movie Reviews
Bag-of-Words Dense Autoencoder

Author: Stephen Pasco
Course: CSCI E-89b NLP
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings('ignore')

def main():
    np.random.seed(42)

    print("="*70)
    print("IMDb BAG-OF-WORDS AUTOENCODER")
    print("="*70)

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
        """Convert bag-of-words to word list"""
        if for_reconstructed:
            # For reconstructions: show top-K words by probability
            top_indices = np.argsort(bow_vector)[::-1][:top_k]
            words = [index_to_word.get(int(idx), '<UNK>') for idx in sorted(top_indices)]
        else:
            # For originals: show words that are present (value = 1.0)
            word_indices = np.where(bow_vector > 0.5)[0]
            words = [index_to_word.get(int(idx), '<UNK>') for idx in sorted(word_indices)[:top_k]]
        return words

    # ============================================================================
    # BUILD AND TRAIN AUTOENCODERS
    # ============================================================================
    print("\n[3] Training autoencoders with different coding sizes...")

    coding_sizes = [64, 32, 16]
    results = {}

    for coding_size in coding_sizes:
        print(f"\n  Training {coding_size} codings...", end=' ')

        # Build model
        encoder_input = layers.Input(shape=(max_features,))
        x = layers.Dense(128, activation='relu')(encoder_input)
        x = layers.Dense(64, activation='relu')(x)
        encoder_output = layers.Dense(coding_size, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(encoder_output)
        x = layers.Dense(128, activation='relu')(x)
        decoder_output = layers.Dense(max_features, activation='sigmoid')(x)

        model = models.Model(encoder_input, decoder_output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train
        history = model.fit(
            X_train_bow, X_train_bow,
            epochs=15,
            batch_size=128,
            validation_data=(X_test_bow, X_test_bow),
            verbose=0
        )

        # Store results
        results[coding_size] = {
            'model': model,
            'history': history,
            'val_loss': history.history['val_loss'][-1],
            'val_mae': history.history['val_mae'][-1]
        }

        print(f"✓ MSE: {results[coding_size]['val_loss']:.4f}")

    # ============================================================================
    # PLOT TRAINING CURVES
    # ============================================================================
    print("\n[4] Generating loss curves...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['blue', 'green', 'red']

    for idx, size in enumerate(coding_sizes):
        hist = results[size]['history']
        epochs = range(1, len(hist.history['loss']) + 1)

        ax1.plot(epochs, hist.history['loss'], color=colors[idx],
                 linestyle='-', linewidth=2, label=f'{size} codings (train)')
        ax1.plot(epochs, hist.history['val_loss'], color=colors[idx],
                 linestyle='--', linewidth=2, label=f'{size} codings (val)')

        ax2.plot(epochs, hist.history['mae'], color=colors[idx],
                 linestyle='-', linewidth=2, label=f'{size} codings (train)')
        ax2.plot(epochs, hist.history['val_mae'], color=colors[idx],
                 linestyle='--', linewidth=2, label=f'{size} codings (val)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Training & Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('5/problem3_loss_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 5/problem3_loss_curves.png")
    plt.close()

    # ============================================================================
    # EVALUATE 5 RANDOM SAMPLES (using smallest viable coding size)
    # ============================================================================
    print("\n[5] Demonstrating effectiveness with 5 random test samples...")

    # Determine smallest viable coding size
    mse_threshold = 0.025
    smallest_coding = min([s for s in coding_sizes if results[s]['val_loss'] <= mse_threshold])

    print(f"\nUsing {smallest_coding} codings (smallest with MSE ≤ {mse_threshold}):")
    print(f"Compression: {max_features}:{smallest_coding} = {max_features/smallest_coding:.1f}:1")
    print("="*70)

    # Get 5 random test samples
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test_bow), 5, replace=False)
    test_samples = X_test_bow[random_indices]

    # Reconstruct using smallest coding model
    model = results[smallest_coding]['model']
    reconstructed = model.predict(test_samples, verbose=0)

    # Show each sample
    for i in range(5):
        print(f"\nSample {i+1} (Index: {random_indices[i]}, Label: {y_test[random_indices[i]]})")
        print("-"*70)

        # Decode
        orig_words = decode_bow(test_samples[i], top_k=50, for_reconstructed=False)
        recon_words = decode_bow(reconstructed[i], top_k=50, for_reconstructed=True)

        print(f"ORIGINAL (top 50 words):\n{' '.join(orig_words)}")
        print(f"\nRECONSTRUCTED (top 50 words):\n{' '.join(recon_words)}")

        # Metrics
        mse = np.mean((test_samples[i] - reconstructed[i])**2)
        overlap = len(set(orig_words) & set(recon_words))
        overlap_pct = (overlap / len(orig_words)) * 100 if len(orig_words) > 0 else 0

        print(f"\nMSE: {mse:.4f} | Word Overlap: {overlap}/{len(orig_words)} ({overlap_pct:.1f}%)")

    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    for size in coding_sizes:
        r = results[size]
        print(f"\n{size} codings:")
        print(f"  Val MSE: {r['val_loss']:.4f}")
        print(f"  Val MAE: {r['val_mae']:.4f}")
        print(f"  Compression: {max_features/size:.1f}:1")

    # ============================================================================
    # ANSWER
    # ============================================================================
    print("\n" + "="*70)
    print("ANSWER TO QUESTION [6]")
    print("="*70)
    print(f"\n** The smallest number of codings is {smallest_coding} **")
    print(f"   (Achieves MSE: {results[smallest_coding]['val_loss']:.4f})")
    print(f"   (Compression ratio: {max_features/smallest_coding:.1f}:1)")

    print("\n" + "="*70)
    print("✓ Problem 3 Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Redirect stdout to output file
    output_file = '5/problem3_output.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        # Save original stdout
        original_stdout = sys.stdout
        # Redirect stdout to file
        sys.stdout = f

        try:
            main()
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

    print(f"✓ Output written to: {output_file}")
