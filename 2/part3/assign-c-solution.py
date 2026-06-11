#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assignment 2 Part C: RNN Movie Review Classification with Sequence Processing
Uses LSTM architecture with sequences of word indices and an Embedding layer.
"""

import os
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models, layers
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## 1. Define Parameters and Load Data

# --- Configuration ---
# Number of words to consider as features
max_features = 10000  # Using a larger vocabulary size
# Cut texts after this number of words
maxlen = 500
# Dimension of the word embeddings
embedding_dim = 200  # Changed to 200 to meet the required input shape (samples, 500, 200) for the next layer
# LSTM units
lstm_units = 32  # Using 32 units

# --- Load the IMDB dataset ---
print("Loading data...")
# Load data with the top max_features most frequent words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)
print(len(train_data), 'train sequences')
print(len(test_data), 'test sequences')

## 2. Preprocess the Data into Sequences

# --- Pad sequences to a fixed length ---
# This ensures all input sequences to the network have the same size.
# Shorter reviews are padded with 0s, and longer ones are truncated.
print("\nPreprocessing data (padding sequences)...")
x_train = sequence.pad_sequences(train_data, maxlen=maxlen)
x_test = sequence.pad_sequences(test_data, maxlen=maxlen)

# --- Vectorize labels ---
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

## 3. Build the Recurrent Neural Network (RNN) Model

# --- Define the model architecture ---
# We use an Embedding layer followed by an LSTM layer.
model = models.Sequential()
# 1. Embedding Layer: Turns word indices into dense vectors of size `embedding_dim`.
#    input_dim is max_features + 1 because index 0 is reserved for padding
#    Set output_dim to embedding_dim (which is now 200)
model.add(layers.Embedding(max_features + 1, embedding_dim, input_length=maxlen))
# 2. LSTM Layer: Processes the sequence of vectors to capture temporal patterns.
model.add(layers.LSTM(lstm_units))
# 3. Output Layer: A standard Dense layer for binary classification.
model.add(layers.Dense(1, activation='sigmoid'))

# --- Build the model to show proper summary ---
# This forces Keras to build the layers with the correct shapes
model.build(input_shape=(None, maxlen))

# --- Compile the model ---
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

## 4. Train the Model

# Create a validation set from the training data
val_split = 10000
x_val = x_train[:val_split]
partial_x_train = x_train[val_split:]
y_val = y_train[:val_split]  # Corrected slicing
partial_y_train = y_train[val_split:]  # Corrected slicing

# Use EarlyStopping to find the optimal number of epochs automatically
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop after 3 epochs of no improvement in validation loss
    restore_best_weights=True  # Restore model to the best state found
)

print("\n[INFO] Training model with EarlyStopping...")
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,  # Train for up to 20 epochs; EarlyStopping will stop it sooner
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Determine the optimal number of epochs from the history (where validation loss was lowest)
optimal_epoch_index = np.argmin(history.history['val_loss'])
optimal_epochs = optimal_epoch_index + 1  # Epochs are 1-indexed

print(f"\n[INFO] Training stopped. Optimal number of epochs (based on validation loss): {optimal_epochs}")

## 5. Evaluate the Model

print("\n[INFO] Evaluating model on test set...")
# Evaluate the model on the test set using the weights from the optimal epoch
# EarlyStopping with restore_best_weights=True has already restored the model
# to the state with the best validation loss, which corresponds to the optimal epoch.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy with optimal {optimal_epochs} epochs: {test_acc * 100:.2f}%")

# Check if accuracy target is met
if test_acc >= 0.75:
    print("[SUCCESS] Accuracy target of >= 75% was achieved.")
else:
    print("[NOTE] Accuracy target of >= 75% was not met. Further tuning may be needed.")

## 6. Plot Results

print("\n--- Requirement (a): Plotting Results ---")
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Plot accuracy
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plot_filename_acc = os.path.join(SCRIPT_DIR, 'movie_review_rnn_accuracy.png')
plt.savefig(plot_filename_acc)
print(f"[SUCCESS] Accuracy plot saved to file: {plot_filename_acc}")
plt.close()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plot_filename_loss = os.path.join(SCRIPT_DIR, 'movie_review_rnn_loss.png')
plt.savefig(plot_filename_loss)
print(f"[SUCCESS] Loss plot saved to file: {plot_filename_loss}")
plt.close()

## 7. Explain Data Representation

print("\n--- Requirement (c): Explanation of Data Representation ---")

print("\nExplanation for `x_train[0]`:")
print("--------------------------")
print(f"Shape: {x_train[0].shape}")
explanation_0 = """
`x_train[0]` represents a **single, complete movie review** as a sequence
of word indices.

It is a 1D array with a shape of (500,). Each element in this array is an
integer representing a word's index in the vocabulary. This is the input format
expected by the Embedding layer, which will learn to represent each word index
as a dense vector.
"""
print(explanation_0)

print("\nExplanation for `x_train[0,0]`:")
print("----------------------------")
print(f"Value: {x_train[0,0]}")
explanation_0_0 = """
`x_train[0,0]` represents the **index of the very first word of the first movie review**.

It is a single integer value. This integer corresponds to the word's position
in the vocabulary (which includes the top 10000 most frequent words in the IMDB dataset).
The Embedding layer will use this index to look up the corresponding dense vector
representation for this word.
"""
print(explanation_0_0)

## 8. Save Report to File

print("\n[INFO] Generating assignment report...")
report_filename = os.path.join(SCRIPT_DIR, 'assignment2c_report.txt')
with open(report_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ASSIGNMENT 2 PART C: RNN MOVIE REVIEW CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write("Configuration:\n")
    f.write(f"- Vocabulary size: {max_features} words\n")
    f.write(f"- Sequence length: {maxlen} timesteps\n")
    f.write(f"- Embedding dimension: {embedding_dim}\n")
    f.write(f"- LSTM units: {lstm_units}\n\n")
    f.write("Results:\n")
    f.write(f"- Optimal number of epochs: {optimal_epochs}\n")
    f.write(f"- Test accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"- Test loss: {test_loss:.4f}\n\n")
    f.write("Data Representation:\n")
    f.write(f"- x_train[0] shape: {x_train[0].shape} (sequence of word indices)\n")
    f.write(f"- x_train[0,0] value: {x_train[0,0]} (first word index)\n\n")
    f.write(f"Target accuracy (>= 75%): {'ACHIEVED' if test_acc >= 0.75 else 'NOT MET'}\n")
    f.write("\nPlot files generated:\n")
    f.write(f"- {plot_filename_acc}\n")
    f.write(f"- {plot_filename_loss}\n")

print(f"[SUCCESS] Report saved to: {report_filename}")
print("\n" + "=" * 70)
print("Assignment 2 Part C Complete!")
print(f"Test Accuracy: {test_acc * 100:.2f}% ({'PASS' if test_acc >= 0.75 else 'NEEDS IMPROVEMENT'})")
print("=" * 70)