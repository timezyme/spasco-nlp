# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Natural Language Processing course assignments repository containing:
- Jupyter notebooks with deep learning text classification examples
- Assignment submissions and documentation

## Development Environment

### Python Setup
- Python 3.13.2 installed
- Jupyter notebooks are the primary development environment

### Key Dependencies
The example notebook uses Keras for deep learning:
- `keras` - Deep learning framework
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization

## Common Development Tasks

### Running Jupyter Notebooks
```bash
# Install Jupyter if not present
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Installing Dependencies
```bash
# Install required packages for the Reuters classification example
pip install keras tensorflow numpy matplotlib
```

### Working with the Notebooks
- The main example notebook (`1/assign1-question1.ipynb`) demonstrates multi-class text classification using the Reuters dataset
- Contains a complete workflow: data loading, preprocessing, model building, training, and evaluation
- Uses fully connected neural networks for text classification

## Architecture and Concepts

### Text Classification Pipeline
The notebook implements a standard NLP classification pipeline:
1. **Data Loading**: Reuters dataset with 46 topic categories
2. **Text Vectorization**: One-hot encoding of word sequences (vocabulary size: 10,000)
3. **Label Encoding**: Categorical encoding for multi-class classification
4. **Model Architecture**: Sequential Dense layers with ReLU activation and softmax output
5. **Training**: Uses validation split to monitor overfitting

### Key Implementation Details
- **Network Architecture**: Input (10,000) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(46, softmax)
- **Loss Function**: `categorical_crossentropy` for multi-class classification
- **Optimizer**: RMSprop
- **Validation Strategy**: 1,000 sample validation set from training data
- **Overfitting Detection**: Model typically starts overfitting after 8 epochs

## File Structure
```
.
├── 1/
│   ├── assign1-question1.ipynb    # Main classification example notebook
│   └── assign1-stephen-pasco.docx # Assignment submission document
├── example/
│   └── example.pdf                # Example reference material
└── assign1-orig.docx              # Original assignment instructions
```

## Notes for Development
- The notebook demonstrates the importance of avoiding information bottlenecks in neural networks (e.g., using 4-dimensional hidden layers significantly reduces accuracy)
- Experiments suggested in the notebook include varying layer sizes (32, 128 units) and number of hidden layers (1 or 3 instead of 2)
- The baseline random accuracy for this 46-class problem is ~19%, and the model achieves ~78% accuracy