# CSCI-E-89b Introduction to Natural Language Processing - Assignments

## Project Overview

This repository contains coursework for Harvard Extension School's CSCI-E-89b Introduction to Natural Language Processing course. The assignments progress from foundational NLP concepts to advanced machine learning and deep learning applications.

---

## Environment Setup

### Python Virtual Environment
- **Location:** `venv/` (project root)
- **Python Version:** 3.13.7
- **Activation:** `source venv/bin/activate`
- **Required for:** All assignments (especially assignments 2-8)

### Key Libraries
- **Deep Learning:** Keras 3.x (standalone), TensorFlow
- **NLP:** NLTK, spaCy
- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Jupyter:** notebook, nbconvert

**Important:** Always use `venv/bin/python` for running scripts to avoid environment conflicts.

---

## Assignment Progress

### ✅ Assignment 1: Foundations
**Status:** Completed
**Topics:** Basic NLP concepts, text processing fundamentals
**Key Files:**
- `1/assign1-question1.ipynb` - Main notebook
- `1/assign1-stephen-pasco.docx` - Final deliverable

---

### ✅ Assignment 2: Text Classification Basics
**Status:** Completed
**Topics:** Movie review classification, sentiment analysis
**Key Files:**
- `2/3.5-classifying-movie-reviews.ipynb` - Implementation
- `2/Assign2-StephenPasco.docx` - Final deliverable
- `2/README.md` - Lessons learned (environment setup)

**Critical Learning:** Environment setup and Keras import patterns

---

### ✅ Assignment 3: Advanced Text Processing
**Status:** Completed
**Topics:** Text preprocessing, feature engineering
**Key Files:**
- `3/CSCI_S-89B_Assignment3.docx` - Assignment instructions

---

### ✅ Assignment 4: Model Evaluation
**Status:** Completed
**Topics:** Model evaluation metrics, performance analysis
**Key Files:**
- `4/3.5-classifying-movie-reviews.ipynb` - Implementation
- `4/StephenPasco-Assign4.docx` - Final deliverable
- `4/test_accuracy_90plus.txt` - Performance achievements

---

### ✅ Assignment 5: Character Embeddings & Autoencoders
**Status:** Completed
**Topics:** Character-level embeddings, sequence modeling, autoencoders
**Key Files:**
- `5/StephenPasco-Assignment5.ipynb` - Main notebook
- `5/problem1/` - Character embedding solutions
- `5/problem2/` - Text generation with RNNs
- `5/problem3/` - Autoencoder for IMDb reviews
- `5/PROBLEM3_SOLUTION_MEMORY.md` - Technical documentation

**Highlights:**
- Character-level embedding with unknown token handling
- LSTM-based character prediction models
- Autoencoder architecture for text representation

---

### ✅ Assignment 6: Advanced Neural Networks
**Status:** Completed
**Topics:** Deep neural network architectures
**Key Files:**
- `6/problem-1-2-3/` - Problem implementations
- `6/Assign6-StephenPasco.docx` - Final deliverable

---

### ✅ Assignment 7: R-based Analysis
**Status:** Completed
**Topics:** R programming for NLP, statistical analysis
**Key Files:**
- `7/Assign7.Rmd` - R Markdown source
- `7/Assign7.html` - HTML report
- `7/Assign7-StephenPasco.docx` - Final deliverable
- `7/data/` - Analysis datasets

---

### 🔄 Assignment 8: Text Classification with Machine Learning
**Status:** In Progress - Problem 1 Complete
**Topics:** SMS spam classification, ML algorithm comparison, neural networks

#### Completed ✅
**Problem 1: Traditional ML Classifiers** (All parts a-g)
- Data preparation with TF-IDF vectorization
- Naive Bayes, KNN, Logistic Regression implementations
- SVM with hyperparameter tuning (kernel, C parameter)
- Random Forest with parameter analysis
- Comprehensive performance comparison visualizations

**Recent Commits:**
- `2a58cb2` - Problem 1(g): Model performance visualization and analysis
- `f15b7c6` - Problem 1(f): Random Forest classifier
- `f889092` - Problem 1(e): SVM with grid search
- `52f6e1b` - Problem 1(d): Logistic Regression
- `1fe84f4` - Problem 1(c): KNN with parameter tuning

#### In Progress 🔄
**Problem 2: Neural Network Classifier**
- Target: Test accuracy ≥ 0.98
- Custom preprocessing and feature engineering required
- Will include updated comparison with Problem 1 models

**Key Files:**
- `8/Assign8.ipynb` - Main implementation notebook
- `8/data/Spam_SMS.csv` - SMS spam dataset
- `8/README.md` - Detailed problem requirements and progress
- `8/TA-notes/` - Course materials and tutorials

**Branch:** `problem-8`

---

## Repository Structure

```
spasco-nlp/
├── README.md                     # This file
├── CLAUDE.md                     # AI assistant configuration (local, gitignored)
├── venv/                         # Python virtual environment (create locally, gitignored)
├── 1-9/                          # Assignment directories
│   ├── *.ipynb                   # Jupyter notebooks
│   ├── *.docx                    # Assignment docs & deliverables
│   └── README.md                 # Assignment-specific docs
├── final-project/                # Final project materials
└── guides/                       # Reference materials
```

---

## Development Workflow

### Standard Assignment Process
1. **Setup**: Activate venv - `source venv/bin/activate`
2. **Branch**: Create feature branch - `git checkout -b problem-X`
3. **Implement**: Work in Jupyter notebook
4. **Test**: Validate results and metrics
5. **Document**: Update README with progress
6. **Commit**: Incremental commits with descriptive messages
7. **Merge**: Merge to main when complete

### Current Branch
- **Branch:** `problem-8`
- **Working on:** Assignment 8, Problem 2 (Neural Network)
- **Base branch:** `main`

---

## Key Learnings & Best Practices

### Environment Management
✅ Always use project venv (`venv/bin/python`)
✅ Verify environment before importing libraries
✅ Check for existing venv directories before creating new ones
❌ Never use system Python or conda base for project work

### Import Patterns
✅ Keras 3.x: `from keras import models, layers`
❌ Avoid: `from tensorflow.keras import ...` (compatibility issues)

### Code Organization
- Jupyter notebooks for primary implementations
- `.docx` files for final deliverables
- `README.md` for documentation and progress tracking
- Separate problem directories when multiple problems exist

### Git Practices
- Feature branches for all work
- Incremental, descriptive commits
- Update documentation with progress
- Clean working directory before merging

---

## Technical Stack

### Core Libraries
- **Deep Learning:** Keras 3.x, TensorFlow
- **NLP Processing:** NLTK, spaCy, transformers
- **Machine Learning:** scikit-learn
- **Data Science:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

### Development Tools
- **Jupyter Notebook:** Interactive development
- **Git:** Version control on feature branches
- **VS Code:** Primary editor with Claude Code integration

---

## Resources

### Course Materials
Each assignment directory contains:
- `TA-notes/` - Teaching assistant tutorials and examples
- `data/` - Datasets and training materials
- Assignment instructions (`.docx` files)

### Reference Guides
Located in `guides/`:
- LLMs in the GitHub Ecosystem
- Software Architecture Patterns
- NLP-ML prompt engineering

---

## Contact & Attribution

**Course:** CSCI-E-89b Introduction to Natural Language Processing
**Institution:** Harvard
**Student:** Stephen Pasco
**Academic Term:** Fall 2025

---

## Recent Updates

**Date:** 2025-11-05
**Last Updated By:** Claude Code (via /sc:index)

**Recent Changes:**
- ✅ Completed Assignment 8, Problem 1 (parts a-g)
- 🔄 Started Assignment 8, Problem 2 (neural network implementation)
- 📝 Added comprehensive progress tracking and documentation
- 🔧 Standardized README format across assignments

**Next Steps:**
1. Complete Assignment 8, Problem 2 (neural network with ≥0.98 accuracy)
2. Generate comparison visualizations with all models
3. Finalize Assignment 8 deliverable document
4. Merge `problem-8` branch to main

---

**Generated with:** Claude Code /sc:index command
**Last Documentation Review:** 2025-11-05
