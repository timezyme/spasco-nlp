#!/usr/bin/env python
"""
CNN for Handwritten Digit Classification with Accuracy Plotting
Based on the example from Deep Learning with Python, Chapter 5.1
Enhanced with best practices for reproducibility and evaluation
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Keras imports
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    # Ensure reproducibility for GPU operations
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_cnn_model():
    """Build the CNN model architecture as specified in the example"""
    model = models.Sequential()

    # Add Input layer as recommended by Keras to avoid warning
    model.add(layers.Input(shape=(28, 28, 1)))

    # Conv2D + MaxPooling layers with Batch Normalization
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    # Flatten and Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model

def prepare_data():
    """Load and preprocess the MNIST dataset"""
    # Load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape to add channel dimension
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Convert labels to categorical (one-hot encoding)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

def plot_training_history(history, output_dir):
    """Plot training and validation accuracy versus epochs with error handling"""
    try:
        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Model Accuracy vs Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], 'b-', label='Training Loss')
        plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Model Loss vs Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot using relative path
        output_path = os.path.join(output_dir, 'part3_training_history.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\nPlot saved successfully to {output_path}")

    except Exception as e:
        print(f"\nWarning: Error saving plot: {e}")
        print("Displaying plot instead...")

    plt.show()

def plot_confusion_matrix(model, test_images, test_labels, output_dir):
    """Generate and plot confusion matrix"""
    try:
        # Get predictions
        predictions = model.predict(test_images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)

        # Generate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title('Confusion Matrix for MNIST Digit Classification', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        # Save plot
        output_path = os.path.join(output_dir, 'part3_confusion_matrix.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
        plt.show()

    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

def evaluate_model_detailed(model, test_images, test_labels):
    """Generate detailed classification report"""
    try:
        predictions = model.predict(test_images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)

        print("\n" + "=" * 50)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(true_classes, pred_classes,
                                  target_names=[str(i) for i in range(10)]))

        # Calculate per-class accuracy
        print("\nPer-Class Accuracy:")
        for i in range(10):
            mask = true_classes == i
            class_acc = np.mean(pred_classes[mask] == i)
            print(f"  Digit {i}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    except Exception as e:
        print(f"Error generating detailed evaluation: {e}")

def main():
    """Main execution function with enhanced features"""
    # Get output directory (same as script location)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Enhanced CNN for Handwritten Digit Classification")
    print("=" * 50)

    # Set random seeds for reproducibility
    print("\nSetting random seeds for reproducibility...")
    set_seeds(42)

    # Prepare data
    print("\n1. Loading and preprocessing MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = prepare_data()
    print(f"   Training samples: {len(train_images)}")
    print(f"   Test samples: {len(test_images)}")

    # Build model
    print("\n2. Building enhanced CNN model with batch normalization and dropout...")
    model = build_cnn_model()
    model.summary()

    # Compile model with optimized settings
    print("\n3. Compiling model with RMSprop optimizer...")
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Setup callbacks
    print("\n4. Setting up callbacks (EarlyStopping, ModelCheckpoint, LR Reduction)...")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train model with validation split
    print("\n5. Training model with validation split and callbacks...")
    history = model.fit(train_images, train_labels,
                       epochs=10,
                       batch_size=128,  # Increased for faster training
                       validation_split=0.2,
                       callbacks=callbacks,
                       verbose=1)

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Plot training history
    print("\n7. Plotting training and validation metrics...")
    plot_training_history(history, output_dir)

    # Generate confusion matrix
    print("\n8. Generating confusion matrix...")
    plot_confusion_matrix(model, test_images, test_labels, output_dir)

    # Detailed evaluation
    print("\n9. Generating detailed classification report...")
    evaluate_model_detailed(model, test_images, test_labels)

    # Save final model
    print("\n10. Saving final model...")
    final_model_path = os.path.join(output_dir, 'part3_cnn_model.keras')
    model.save(final_model_path)
    print(f"    Model saved to {final_model_path}")

    # Print final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Total epochs trained: {len(history.history['accuracy'])}")

    return model, history

if __name__ == "__main__":
    model, history = main()