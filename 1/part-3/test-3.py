"""
Optimizer Experiments for Reuters Text Classification
Testing various optimizers and hyperparameters to find optimal configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop, AdamW, Nadam, Adamax
from keras.callbacks import EarlyStopping
import keras
from datetime import datetime
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
import os
import sys
from io import StringIO


class TeeOutput:
    """Class to duplicate output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class OptimizerExperiment:
    """
    Run experiments with different optimizers and hyperparameters
    """
    
    def __init__(self, num_words=10000, num_classes=46, seed=42):
        self.num_words = num_words
        self.num_classes = num_classes
        self.seed = seed
        keras.utils.set_random_seed(seed)
        
        # Load and prepare data once
        self.load_data()
        
        # Store results
        self.results = []
        
    def load_data(self):
        """Load and prepare Reuters dataset"""
        print("Loading Reuters dataset...")
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
            num_words=self.num_words
        )
        
        # Vectorize sequences
        self.x_train_full = self._vectorize_sequences(train_data)
        self.x_test = self._vectorize_sequences(test_data)
        
        # One-hot encode labels
        one_hot_train_labels = to_categorical(train_labels, self.num_classes)
        one_hot_test_labels = to_categorical(test_labels, self.num_classes)
        
        # Create validation set
        self.x_val = self.x_train_full[:1000]
        self.x_train = self.x_train_full[1000:]
        
        self.y_val = one_hot_train_labels[:1000]
        self.y_train = one_hot_train_labels[1000:]
        
        self.y_test = one_hot_test_labels
        
        print(f"Data prepared: Train={len(self.x_train)}, Val={len(self.x_val)}, Test={len(self.x_test)}")
        
    def _vectorize_sequences(self, sequences):
        """Convert sequences to binary vectors"""
        results = np.zeros((len(sequences), self.num_words), dtype='float32')
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    
    def build_model(self, optimizer):
        """Build the model with specified optimizer"""
        model = models.Sequential()
        
        # Using improved architecture from part-3.py
        model.add(layers.Dense(256, input_shape=(self.num_words,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.4))
        
        model.add(layers.Dense(128))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(64))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def run_experiment(self, optimizer, optimizer_name, hyperparams_str, epochs=30, batch_size=256):
        """Run a single experiment with given optimizer"""
        print(f"\n{'='*60}")
        print(f"Testing: {optimizer_name} - {hyperparams_str}")
        print(f"{'='*60}")
        
        # Reset random seed for reproducibility
        keras.utils.set_random_seed(self.seed)
        
        # Build and train model
        model = self.build_model(optimizer)
        
        # Use early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get best validation accuracy
        val_acc_history = history.history['val_accuracy']
        best_val_acc = max(val_acc_history)
        best_epoch = val_acc_history.index(best_val_acc) + 1
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Store results
        result = {
            'optimizer': optimizer_name,
            'hyperparameters': hyperparams_str,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'test_accuracy': test_acc,
            'history': history.history
        }
        
        self.results.append(result)
        
        print(f"\nResults:")
        print(f"  Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        return result
    
    def run_all_experiments(self):
        """Run experiments with various optimizers"""
        
        experiments = [
            # TOP 5 OPTIMIZERS ONLY - Based on validation accuracy performance
            
            # RMSprop - Best overall performer (0.8280 val acc)
            (RMSprop(learning_rate=0.001), "RMSprop", "lr=0.001"),
            (RMSprop(learning_rate=0.0001), "RMSprop", "lr=0.0001"),
            (RMSprop(learning_rate=0.001, rho=0.95), "RMSprop", "lr=0.001, rho=0.95"),
            (RMSprop(learning_rate=0.01, rho=0.9), "RMSprop", "lr=0.01, rho=0.9"),
            
            # Adam - Second best (0.8270 val acc)
            (Adam(learning_rate=0.001), "Adam", "lr=0.001"),
            (Adam(learning_rate=0.0001), "Adam", "lr=0.0001"),
            (Adam(learning_rate=0.01), "Adam", "lr=0.01"),
            (Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), "Adam", "lr=0.001, β1=0.9, β2=0.999"),
            
            # Adamax - Tied second (0.8270 val acc)
            (Adamax(learning_rate=0.002), "Adamax", "lr=0.002"),
            
            # AdamW - Fourth best (0.8250 val acc)
            (AdamW(learning_rate=0.001, weight_decay=0.01), "AdamW", "lr=0.001, wd=0.01"),
            (AdamW(learning_rate=0.001, weight_decay=0.001), "AdamW", "lr=0.001, wd=0.001"),
            (AdamW(learning_rate=0.0001, weight_decay=0.01), "AdamW", "lr=0.0001, wd=0.01"),
            
            # Nadam - Fifth best (0.8220 val acc)
            (Nadam(learning_rate=0.001), "Nadam", "lr=0.001"),
            (Nadam(learning_rate=0.002), "Nadam", "lr=0.002"),
        ]
        
        # Run each experiment
        for optimizer, name, params in experiments:
            self.run_experiment(optimizer, name, params)
    
    def plot_results(self):
        """Create comprehensive visualization of results"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Training curves for best performers from each optimizer family
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        
        # Group results by optimizer type
        optimizer_families = {}
        for result in self.results:
            base_name = result['optimizer'].split()[0]  # Get base optimizer name
            if base_name not in optimizer_families:
                optimizer_families[base_name] = []
            optimizer_families[base_name].append(result)
        
        # Plot best from each family
        colors = plt.cm.tab10(np.linspace(0, 1, len(optimizer_families)))
        
        for idx, (family, family_results) in enumerate(optimizer_families.items()):
            # Find best performer in family
            best = max(family_results, key=lambda x: x['best_val_accuracy'])
            
            epochs = range(1, len(best['history']['accuracy']) + 1)
            
            # Plot training accuracy
            ax1.plot(epochs, best['history']['accuracy'], 
                    label=f"{best['optimizer']} ({best['hyperparameters']})",
                    color=colors[idx], linestyle='-', alpha=0.7)
            
            # Plot validation accuracy
            ax2.plot(epochs, best['history']['val_accuracy'],
                    label=f"{best['optimizer']} ({best['hyperparameters']})",
                    color=colors[idx], linestyle='-', alpha=0.7)
        
        ax1.set_title('Training Accuracy - Best from Each Optimizer Family')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend(fontsize=8, loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Validation Accuracy - Best from Each Optimizer Family')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend(fontsize=8, loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Bar chart comparing best validation accuracies
        ax3 = plt.subplot(2, 3, 3)
        
        # Sort results by validation accuracy - show all configurations
        sorted_results = sorted(self.results, key=lambda x: x['best_val_accuracy'], reverse=True)
        
        labels = [f"{r['optimizer']}\n{r['hyperparameters']}" for r in sorted_results]
        values = [r['best_val_accuracy'] for r in sorted_results]
        
        bars = ax3.barh(range(len(values)), values)
        ax3.set_yticks(range(len(values)))
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel('Best Validation Accuracy')
        ax3.set_title('All Optimizer Configurations (Ranked by Validation Accuracy)')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Color bars by optimizer type
        for i, bar in enumerate(bars):
            optimizer_type = sorted_results[i]['optimizer'].split()[0]
            color_idx = list(optimizer_families.keys()).index(optimizer_type)
            bar.set_color(colors[color_idx])
        
        # 4. Learning rate comparison for Adam
        ax4 = plt.subplot(2, 3, 4)
        
        adam_results = [r for r in self.results if 'Adam' in r['optimizer'] and 'AdamW' not in r['optimizer']]
        for result in adam_results:
            epochs = range(1, len(result['history']['val_accuracy']) + 1)
            ax4.plot(epochs, result['history']['val_accuracy'],
                    label=result['hyperparameters'], alpha=0.7)
        
        ax4.set_title('Adam Optimizer - Learning Rate Comparison')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. AdamW weight decay comparison
        ax5 = plt.subplot(2, 3, 5)
        
        adamw_results = [r for r in self.results if r['optimizer'] == 'AdamW']
        for result in adamw_results:
            epochs = range(1, len(result['history']['val_accuracy']) + 1)
            ax5.plot(epochs, result['history']['val_accuracy'],
                    label=result['hyperparameters'], alpha=0.7)
        
        ax5.set_title('AdamW Optimizer - Weight Decay Comparison')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Validation Accuracy')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Convergence speed comparison
        ax6 = plt.subplot(2, 3, 6)
        
        for family, family_results in optimizer_families.items():
            best = max(family_results, key=lambda x: x['best_val_accuracy'])
            epochs = range(1, len(best['history']['val_accuracy']) + 1)
            ax6.plot(epochs, best['history']['val_accuracy'],
                    label=f"{family} (best config)", alpha=0.7, linewidth=2)
        
        ax6.set_title('Convergence Speed Comparison - Best Configurations')
        ax6.set_xlabel('Epochs')
        ax6.set_ylabel('Validation Accuracy')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Optimizer Experiments - Reuters Text Classification', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, 'optimizer_comparison.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.show()
    
    def generate_results_table(self):
        """Generate and display results table"""
        # Create summary table
        summary = []
        
        # Group by optimizer type
        optimizer_groups = {}
        for result in self.results:
            base_name = result['optimizer']
            if base_name not in optimizer_groups:
                optimizer_groups[base_name] = []
            optimizer_groups[base_name].append(result)
        
        # Find best configuration for each optimizer
        for optimizer, group in optimizer_groups.items():
            best = max(group, key=lambda x: x['best_val_accuracy'])
            summary.append({
                'Optimizer': optimizer,
                'Best Hyperparameters': best['hyperparameters'],
                'Best Val Accuracy': f"{best['best_val_accuracy']:.4f}",
                'Best Epoch': best['best_epoch'],
                'Test Accuracy': f"{best['test_accuracy']:.4f}"
            })
        
        # Sort by best validation accuracy
        summary.sort(key=lambda x: float(x['Best Val Accuracy']), reverse=True)
        
        print("\n" + "="*80)
        print("OPTIMIZER COMPARISON - BEST CONFIGURATIONS")
        print("="*80)
        
        if HAS_PANDAS:
            # Create DataFrame
            df = pd.DataFrame(summary)
            print(df.to_string(index=False))
        else:
            # Print as formatted table without pandas
            # Print header
            print(f"{'Optimizer':<12} {'Best Hyperparameters':<35} {'Best Val Acc':<12} {'Best Epoch':<10} {'Test Acc':<10}")
            print("-"*80)
            for row in summary:
                print(f"{row['Optimizer']:<12} {row['Best Hyperparameters']:<35} {row['Best Val Accuracy']:<12} {row['Best Epoch']:<10} {row['Test Accuracy']:<10}")
        
        # Also save to file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(script_dir, 'optimizer_results.txt')
        
        with open(results_path, 'w') as f:
            f.write("TOP 5 OPTIMIZER EXPERIMENTS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Reuters Text Classification (46 classes)\n")
            f.write(f"Architecture: 256->128->64 units with BatchNorm and Dropout\n")
            f.write(f"Training samples: 7982, Validation: 1000, Test: 2246\n")
            f.write("="*80 + "\n\n")
            
            f.write("SUMMARY TABLE - BEST CONFIGURATION PER OPTIMIZER\n")
            f.write("-"*80 + "\n")
            
            if HAS_PANDAS:
                df = pd.DataFrame(summary)
                f.write(df.to_string(index=False))
            else:
                f.write(f"{'Optimizer':<12} {'Best Hyperparameters':<35} {'Best Val Acc':<12} {'Best Epoch':<10} {'Test Acc':<10}\n")
                f.write("-"*80 + "\n")
                for row in summary:
                    f.write(f"{row['Optimizer']:<12} {row['Best Hyperparameters']:<35} {row['Best Val Accuracy']:<12} {row['Best Epoch']:<10} {row['Test Accuracy']:<10}\n")
            
            f.write("\n\n")
            
            f.write("DETAILED RESULTS - ALL EXPERIMENTS\n")
            f.write("-"*80 + "\n")
            
            for result in sorted(self.results, key=lambda x: x['best_val_accuracy'], reverse=True):
                f.write(f"\nOptimizer: {result['optimizer']}\n")
                f.write(f"Hyperparameters: {result['hyperparameters']}\n")
                f.write(f"Best Validation Accuracy: {result['best_val_accuracy']:.4f} at epoch {result['best_epoch']}\n")
                f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write("-"*40 + "\n")
        
        print(f"\nDetailed results saved to: {results_path}")
        
        return summary if not HAS_PANDAS else df


def main():
    """Run all optimizer experiments"""
    # Set up logging to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(script_dir, f'optimizer_experiments_{timestamp}.log')
    
    # Redirect output to both console and file
    tee = TeeOutput(log_path)
    sys.stdout = tee
    
    try:
        print("="*80)
        print("TOP 5 OPTIMIZER EXPERIMENTS FOR REUTERS TEXT CLASSIFICATION")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Logging to: {log_path}")
        print("="*80)
        
        # Initialize experiment
        experiment = OptimizerExperiment(seed=42)
        
        # Run all experiments
        experiment.run_all_experiments()
        
        # Generate results table
        print("\n" + "="*80)
        print("GENERATING RESULTS...")
        print("="*80)
        
        results_df = experiment.generate_results_table()
        
        # Plot results
        print("\nGenerating comparison plots...")
        experiment.plot_results()
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Print top 3 performers
        print("\nTOP 3 CONFIGURATIONS:")
        top_results = sorted(experiment.results, key=lambda x: x['best_val_accuracy'], reverse=True)[:3]
        for i, result in enumerate(top_results, 1):
            print(f"{i}. {result['optimizer']} ({result['hyperparameters']}): {result['best_val_accuracy']:.4f}")
        
        print(f"\n✓ Full log saved to: {log_path}")
        print(f"✓ Results table saved to: optimizer_results.txt")
        print(f"✓ Plots saved to: optimizer_comparison.png")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()