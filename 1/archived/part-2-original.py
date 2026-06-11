"""
Reuters Text Classification with Neural Network
Implementation of a fully connected neural network for multi-class text classification
Based on Deep Learning with Python by Francois Chollet
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, regularizers
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import keras


class ReutersTextClassifier:
    """
    A neural network classifier for the Reuters dataset with 46 topic categories.
    Implements training with optimal epoch detection and visualization.
    """
    
    def __init__(self, num_words=10000, num_classes=46, seed=None, verbose=1):
        """
        Initialize the Reuters text classifier.
        
        Args:
            num_words: Number of most frequent words to consider (default: 10000)
            num_classes: Number of output classes (default: 46)
            seed: Random seed for reproducibility (default: None)
            verbose: Verbosity level (0=silent, 1=normal, 2=verbose)
        """
        self.num_words = num_words
        self.num_classes = num_classes
        self.verbose = verbose
        self.model = None
        self.history = None
        self.optimal_epochs = None
        
        # Set random seed for reproducibility
        if seed is not None:
            keras.utils.set_random_seed(seed)
        
        # Data placeholders
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """
        Load the Reuters dataset and prepare it for training.
        Creates training (7982 samples), validation (1000 samples), and test sets.
        """
        # Load the Reuters dataset
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
            num_words=self.num_words
        )
        
        # Vectorize the sequences
        self.x_train = self._vectorize_sequences(train_data)
        self.x_test = self._vectorize_sequences(test_data)
        
        # Convert labels to one-hot encoding
        one_hot_train_labels = to_categorical(train_labels, self.num_classes)
        one_hot_test_labels = to_categorical(test_labels, self.num_classes)
        
        # Create validation set from first 1000 training samples
        self.x_val = self.x_train[:1000]
        self.x_train = self.x_train[1000:]  # Remaining 7982 samples for training
        
        self.y_val = one_hot_train_labels[:1000]
        self.y_train = one_hot_train_labels[1000:]
        
        self.y_test = one_hot_test_labels
        
        if self.verbose >= 1:
            print(f"Data prepared:")
            print(f"  Training samples: {len(self.x_train)}")
            print(f"  Validation samples: {len(self.x_val)}")
            print(f"  Test samples: {len(self.x_test)}")
        
    def _vectorize_sequences(self, sequences):
        """
        Convert sequences of word indices to binary vectors.
        
        Args:
            sequences: List of sequences (word indices)
            
        Returns:
            Binary matrix representation of the sequences
        """
        results = np.zeros((len(sequences), self.num_words), dtype='float32')
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    
    def build_model(self, use_regularization=True, architecture='improved'):
        """
        Build the neural network architecture.
        
        Args:
            use_regularization: Whether to add regularization (default: True)
            architecture: Model architecture type ('original', 'improved', 'deep')
        """
        self.model = models.Sequential()
        
        if architecture == 'improved':
            # Improved architecture with better capacity and regularization
            self.model.add(layers.Dense(256, input_shape=(self.num_words,)))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.4))
            
            self.model.add(layers.Dense(128))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.3))
            
            self.model.add(layers.Dense(64))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.2))
            
        elif architecture == 'deep':
            # Deeper architecture for complex patterns
            self.model.add(layers.Dense(512, input_shape=(self.num_words,)))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.5))
            
            self.model.add(layers.Dense(256))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.4))
            
            self.model.add(layers.Dense(128))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.3))
            
            self.model.add(layers.Dense(64))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.2))
            
        else:  # original
            if use_regularization:
                self.model.add(layers.Dense(64, activation='relu', input_shape=(self.num_words,),
                                           kernel_regularizer=regularizers.l2(0.001)))
                self.model.add(layers.Dropout(0.5))
                self.model.add(layers.Dense(64, activation='relu',
                                           kernel_regularizer=regularizers.l2(0.001)))
                self.model.add(layers.Dropout(0.5))
            else:
                self.model.add(layers.Dense(64, activation='relu', input_shape=(self.num_words,)))
                self.model.add(layers.Dense(64, activation='relu'))
        
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if self.verbose >= 1:
            print(f"Model architecture built: {architecture}")
            self.model.summary()
        
    def train(self, epochs=30, batch_size=256, use_early_stopping=True, use_lr_reduction=True):
        """
        Train the model and identify the optimal number of epochs.
        
        Args:
            epochs: Maximum number of epochs to train (default: 30)
            batch_size: Batch size for training (default: 256)
            use_early_stopping: Whether to use early stopping (default: True)
            use_lr_reduction: Whether to use learning rate reduction (default: True)
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        if self.verbose >= 1:
            print(f"\nTraining model for up to {epochs} epochs...")
        
        callbacks = []
        
        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=self.verbose,
                mode='max'
            )
            callbacks.append(early_stopping)
        
        if use_lr_reduction:
            lr_reduction = ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                verbose=self.verbose,
                factor=0.5,
                min_lr=0.00001
            )
            callbacks.append(lr_reduction)
        
        # Add model checkpoint to save best model
        checkpoint = ModelCheckpoint(
            'best_model_weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0
        )
        callbacks.append(checkpoint)
        
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        # Load best weights
        try:
            self.model.load_weights('best_model_weights.h5')
            if self.verbose >= 1:
                print("\nLoaded best model weights from checkpoint")
        except:
            pass
        
        # Identify optimal epochs based on validation accuracy
        self._identify_optimal_epochs()
        
        return self.history
    
    def _identify_optimal_epochs(self):
        """
        Identify the optimal number of epochs based on validation accuracy.
        The optimal epoch is where validation accuracy peaks before overfitting begins.
        """
        # Handle different key names in different Keras versions
        if 'val_accuracy' in self.history.history:
            val_acc = self.history.history['val_accuracy']
        else:
            val_acc = self.history.history['val_acc']
        
        # Find the epoch with maximum validation accuracy
        self.optimal_epochs = np.argmax(val_acc) + 1  # +1 because epochs are 1-indexed
        
        if self.verbose >= 1:
            print(f"\nOptimal number of epochs: {self.optimal_epochs}")
            print(f"Maximum validation accuracy: {max(val_acc):.4f} at epoch {self.optimal_epochs}")
            
            # Check for overfitting
            if self.optimal_epochs < len(val_acc):
                print(f"Note: Model starts overfitting after epoch {self.optimal_epochs}")
            
    def plot_training_history(self, save_path=None, show=True, title_suffix=''):
        """
        Plot training and validation accuracy versus number of epochs.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot (default: True)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        # Handle different key names in different Keras versions
        if 'accuracy' in self.history.history:
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
        else:
            acc = self.history.history['acc']
            val_acc = self.history.history['val_acc']
        
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(epochs, acc, 'bo-', label='Training accuracy', markersize=4)
        ax1.plot(epochs, val_acc, 'r^-', label='Validation accuracy', markersize=4)
        
        # Mark optimal epoch
        if self.optimal_epochs:
            ax1.axvline(x=self.optimal_epochs, color='green', linestyle='--', 
                       label=f'Optimal epoch ({self.optimal_epochs})')
            ax1.plot(self.optimal_epochs, val_acc[self.optimal_epochs-1], 
                    'g*', markersize=15, label=f'Max val acc: {val_acc[self.optimal_epochs-1]:.4f}')
        
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(epochs, loss, 'bo-', label='Training loss', markersize=4)
        ax2.plot(epochs, val_loss, 'r^-', label='Validation loss', markersize=4)
        
        # Mark optimal epoch
        if self.optimal_epochs:
            ax2.axvline(x=self.optimal_epochs, color='green', linestyle='--', 
                       label=f'Optimal epoch ({self.optimal_epochs})')
        
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Reuters Text Classification - Training History{title_suffix}', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            if self.verbose >= 1:
                print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def retrain_with_optimal_epochs(self, batch_size=256, architecture='improved'):
        """
        Retrain the model from scratch using the optimal number of epochs.
        
        Args:
            batch_size: Batch size for training (default: 512)
            
        Returns:
            Tuple of (test_accuracy, test_loss) after training with optimal epochs
        """
        if self.optimal_epochs is None:
            raise ValueError("Optimal epochs not determined. Run train() first.")
        
        if self.verbose >= 1:
            print(f"\nRetraining model with optimal epochs: {self.optimal_epochs}")
        
        # Rebuild the model with improved architecture
        self.build_model(use_regularization=True, architecture=architecture)
        
        # Train with optimal epochs
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.optimal_epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=self.verbose
        )
        
        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        if self.verbose >= 1:
            print(f"\nFinal Results with {self.optimal_epochs} epochs:")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
        
        return test_acc, test_loss
    
    def get_random_baseline(self):
        """
        Calculate the random baseline accuracy for comparison.
        
        Returns:
            Random baseline accuracy
        """
        if self.y_test is None:
            raise ValueError("Test data not loaded. Call load_and_prepare_data() first.")
        
        # Get original labels from one-hot encoding
        test_labels = np.argmax(self.y_test, axis=1)
        
        # Create random predictions
        test_labels_copy = test_labels.copy()
        np.random.shuffle(test_labels_copy)
        
        # Calculate accuracy
        baseline = float(np.sum(test_labels == test_labels_copy)) / len(test_labels)
        
        if self.verbose >= 1:
            print(f"Random baseline accuracy: {baseline:.4f}")
        return baseline


def main():
    """
    Main function to demonstrate the Reuters text classifier.
    """
    print("=" * 60)
    print("Reuters Text Classification with Neural Network")
    print("=" * 60)
    
    # Initialize classifier with reproducible seed
    classifier = ReutersTextClassifier(num_words=10000, num_classes=46, seed=42, verbose=1)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    classifier.load_and_prepare_data()
    
    # Build model
    print("\n2. Building model architecture...")
    classifier.build_model()
    
    # Train model for 20 epochs
    print("\n3. Training model to identify optimal epochs...")
    classifier.train(epochs=20, batch_size=256)
    
    # Plot training history
    print("\n4. Plotting training history...")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'results-plot.png')
    classifier.plot_training_history(save_path=plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Get random baseline for comparison
    print("\n5. Computing random baseline...")
    classifier.get_random_baseline()
    
    # Retrain with optimal epochs
    print("\n6. Retraining with optimal number of epochs...")
    test_accuracy, test_loss = classifier.retrain_with_optimal_epochs(batch_size=256)
    
    # Save Part 1b results to file
    results_path = os.path.join(script_dir, 'part1b_results.txt')
    with open(results_path, 'w') as f:
        f.write("PART 1B - TEST ACCURACY RESULTS\n")
        f.write("================================\n\n")
        f.write("Configuration:\n")
        f.write(f"  Random Seed: 42\n")
        f.write(f"  Dataset: Reuters (46 classes)\n")
        f.write(f"  Vocabulary Size: 10,000 words\n")
        f.write(f"  Training Set Size: {len(classifier.x_train)} samples\n")
        f.write(f"  Validation Set Size: {len(classifier.x_val)} samples\n")
        f.write(f"  Test Set Size: {len(classifier.x_test)} samples\n\n")
        
        f.write("Model Selection:\n")
        f.write(f"  Selection Criterion: Maximum Validation Accuracy\n")
        f.write(f"  Optimal Epochs: {classifier.optimal_epochs}\n")
        
        # Get validation accuracy and loss at optimal epoch
        if 'val_accuracy' in classifier.history.history:
            val_acc_optimal = classifier.history.history['val_accuracy'][classifier.optimal_epochs - 1]
            val_loss_optimal = classifier.history.history['val_loss'][classifier.optimal_epochs - 1]
            val_losses = classifier.history.history['val_loss']
        else:
            val_acc_optimal = classifier.history.history['val_acc'][classifier.optimal_epochs - 1]
            val_loss_optimal = classifier.history.history['val_loss'][classifier.optimal_epochs - 1]
            val_losses = classifier.history.history['val_loss']
        
        # Find epoch with minimum validation loss
        min_val_loss_epoch = np.argmin(val_losses) + 1
        min_val_loss = min(val_losses)
        
        f.write(f"  Validation Accuracy at Optimal Epoch: {val_acc_optimal:.4f}\n")
        f.write(f"  Validation Loss at Optimal Epoch: {val_loss_optimal:.4f}\n")
        f.write(f"  Minimum Validation Loss Epoch: {min_val_loss_epoch} (loss: {min_val_loss:.4f})\n\n")
        
        f.write("Test Set Performance:\n")
        f.write(f"  TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"  TEST LOSS: {test_loss:.4f}\n\n")
        
        # Get baseline accuracy
        baseline = classifier.get_random_baseline()
        f.write("Comparison:\n")
        f.write(f"  Random Baseline: {baseline:.4f} ({baseline*100:.2f}%)\n")
        f.write(f"  Improvement over Baseline: {(test_accuracy - baseline)*100:.2f} percentage points\n")
    
    print(f"Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Optimal epochs identified: {classifier.optimal_epochs}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"✓ Part 1b results saved to: {results_path}")
    print(f"✓ Training plot saved to: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()