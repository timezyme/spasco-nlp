#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 2 Part B: Movie Review Classification with 200-word vocabulary
Modified from the IMDB example to use only the top 200 most frequent words.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting to file
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from datetime import datetime

def vectorize_sequences(sequences, dimension=200):
    """Convert sequences of word indices to binary vectors."""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # Only indices < dimension will be set (words in top 200)
        valid_indices = [idx for idx in sequence if idx < dimension]
        results[i, valid_indices] = 1.
    return results

def generate_report(filename, content):
    """Generate a text report with the results."""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"\nReport generated: {filename}")

def main():
    """Main function for movie review classification with 200-word vocabulary."""

    # Initialize report
    report = []
    report.append("=" * 70)
    report.append("PROJECT 2 PART B: MOVIE REVIEW CLASSIFICATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("\nTask: Movie review sentiment classification using only top 200 words")
    report.append("(Modified from original IMDB example that used 10,000 words)\n")

    print("=" * 70)
    print("Project 2 Part B: Movie Review Classification")
    print("Using only top 200 most frequent words (reduced from 10,000)")
    print("=" * 70)

    # Load IMDB data with num_words=200
    print("\nLoading IMDB dataset with num_words=200...")
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=200)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    report.append("\n1. DATA LOADING")
    report.append("-" * 40)
    report.append(f"Training samples: {len(train_data)}")
    report.append(f"Test samples: {len(test_data)}")
    report.append("Vocabulary size: 200 (reduced from 10,000)")

    # Vectorize the data
    print("\nVectorizing sequences (dimension=200)...")
    x_train = vectorize_sequences(train_data, dimension=200)
    x_test = vectorize_sequences(test_data, dimension=200)

    # Vectorize labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Create validation set
    print("Creating validation set...")
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # Build model
    print("\nBuilding model...")
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(200,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("\nTraining model for 6 epochs...")
    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=6,
        batch_size=512,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Analyze results
    history_dict = history.history

    # Find optimal epochs based on validation loss
    optimal_epochs = np.argmin(history_dict['val_loss']) + 1
    print(f"\nOptimal number of epochs: {optimal_epochs}")
    print(f"Validation accuracy at optimal epoch: {history_dict['val_accuracy'][optimal_epochs-1]:.4f}")

    # Add training results to report
    report.append("\n2. TRAINING RESULTS")
    report.append("-" * 40)
    report.append(f"Epochs trained: 6")
    report.append(f"Final training accuracy: {history_dict['accuracy'][-1]*100:.2f}%")
    report.append(f"Final validation accuracy: {history_dict['val_accuracy'][-1]*100:.2f}%")
    report.append(f"Optimal number of epochs: {optimal_epochs}")
    report.append(f"Validation accuracy at optimal epoch: {history_dict['val_accuracy'][optimal_epochs-1]*100:.2f}%")

    # Part (a): Plot the results TO FILE
    print("\n(a) Creating plot...")

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = 'movie_review_accuracy_200words.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to: {plot_filename}")

    report.append("\n3. RESULTS")
    report.append("=" * 40)
    report.append(f"\n(a) Plot saved: {plot_filename}")

    # Part (b): Train final model with optimal epochs and report test accuracy
    print(f"\n(b) Training final model with {optimal_epochs} optimal epochs...")

    final_model = models.Sequential()
    final_model.add(layers.Dense(16, activation='relu', input_shape=(200,)))
    final_model.add(layers.Dense(16, activation='relu'))
    final_model.add(layers.Dense(1, activation='sigmoid'))

    final_model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train on full training set with optimal epochs
    final_model.fit(x_train, y_train, epochs=optimal_epochs, batch_size=512, verbose=0)

    # Evaluate on test set
    test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy with optimal {optimal_epochs} epochs: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test loss: {test_loss:.4f}")

    report.append(f"\n(b) Test accuracy with optimal {optimal_epochs} epochs: {test_acc*100:.2f}%")
    report.append(f"    Test loss: {test_loss:.4f}")

    # Part (c): Explain what x_train[0] represents
    print("\n(c) Explanation of x_train[0]:")
    print("-" * 40)
    print("x_train[0] represents a single movie review after being processed and vectorized.")
    print("It is a 200-dimensional binary vector where:")
    print("- Each dimension corresponds to one of the top 200 most frequent words")
    print("- A value of 1 indicates the word is present in the review")
    print("- A value of 0 indicates the word is absent from the review")
    print("- This is a 'bag of words' representation (word order is lost)")

    print(f"\nExample analysis of x_train[0]:")
    print(f"  Shape: {x_train[0].shape}")
    print(f"  Number of words present (1s): {int(np.sum(x_train[0]))}")
    print(f"  Number of words absent (0s): {int(200 - np.sum(x_train[0]))}")

    word_indices = np.where(x_train[0] == 1)[0][:10]
    print(f"  First 10 word indices present: {word_indices.tolist()}")

    report.append("\n(c) Explanation of x_train[0]:")
    report.append("-" * 40)
    report.append("x_train[0] is a 200-dimensional binary vector representing a single movie review.")
    report.append(f"- Shape: {x_train[0].shape}")
    report.append(f"- Number of words present (1s): {int(np.sum(x_train[0]))}")
    report.append(f"- Number of words absent (0s): {int(200 - np.sum(x_train[0]))}")
    report.append("- Each dimension corresponds to one of the top 200 most frequent words")
    report.append("- This is a 'bag of words' representation where word order is lost")

    # Add comparison to original
    report.append("\n4. COMPARISON WITH ORIGINAL")
    report.append("-" * 40)
    report.append("Original accuracy (10,000 words): ~88%")
    report.append(f"This implementation (200 words): {test_acc*100:.2f}%")
    report.append(f"Accuracy reduction: ~{88 - test_acc*100:.1f}%")
    report.append("\nConclusion: With only 2% of the original vocabulary,")
    report.append(f"we achieve approximately {test_acc*100/88*100:.1f}% of the original accuracy.")

    # Generate the report file
    report_filename = "project2b_report.txt"
    generate_report(report_filename, "\n".join(report))

    print("\n" + "=" * 70)
    print("Project 2 Part B Complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()