"""
Test script for the Reuters Text Classifier
"""

import tempfile
import os
from reuters_text_classifier import ReutersTextClassifier

def test_classifier():
    """
    Test the Reuters text classifier with a smaller number of epochs for quick validation.
    """
    print("Testing Reuters Text Classifier")
    print("-" * 40)
    
    # Initialize classifier with seed for reproducibility and minimal verbosity
    classifier = ReutersTextClassifier(num_words=10000, num_classes=46, seed=42, verbose=0)
    
    # Load data
    print("Loading data...")
    classifier.load_and_prepare_data()
    
    # Build model
    print("\nBuilding model...")
    classifier.build_model()
    
    # Train for fewer epochs for testing
    print("\nTraining for 5 epochs (quick test)...")
    history = classifier.train(epochs=5, batch_size=512)
    
    # Check that history was recorded
    assert history is not None, "Training history should not be None"
    assert ('accuracy' in history.history or 'acc' in history.history), "Training accuracy should be recorded"
    assert ('val_accuracy' in history.history or 'val_acc' in history.history), "Validation accuracy should be recorded"
    
    # Check optimal epochs were identified
    assert classifier.optimal_epochs is not None, "Optimal epochs should be identified"
    assert 1 <= classifier.optimal_epochs <= 5, "Optimal epochs should be within training range"
    
    print(f"\nTest passed! Optimal epochs identified: {classifier.optimal_epochs}")
    
    # Test plotting (without display) to a temp file
    print("\nGenerating plot...")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    
    fig = classifier.plot_training_history(save_path=temp_path, show=False)
    assert fig is not None, "Figure should be returned"
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Optional: Test that training loss decreases
    history = classifier.history.history
    if 'loss' in history:
        training_losses = history['loss']
        assert training_losses[-1] < training_losses[0], "Training loss should decrease"
        print("✓ Training loss decreased from {:.4f} to {:.4f}".format(training_losses[0], training_losses[-1]))
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_classifier()