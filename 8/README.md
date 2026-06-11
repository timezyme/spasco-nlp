# Project 8: Text Classification with Machine Learning

## Overview

This project covers SMS spam classification using various machine learning algorithms and neural networks. It compares traditional ML classifiers (Naive Bayes, KNN, Logistic Regression, SVM, Random Forest) with neural network approaches to understand model characteristics and performance trade-offs.

---

## Progress Status

### Completed ✅

**Problem 1: Traditional ML Classifiers** (All parts a-g completed)
- ✅ (a) Data Preparation - Dataset loaded, labels converted, train/test split, TF-IDF vectorization
- ✅ (b) Naive Bayes Classifier - Implementation, metrics, confusion matrix visualization
- ✅ (c) k-Nearest Neighbors - Parameter tuning (n_neighbors), metrics, confusion matrix
- ✅ (d) Logistic Regression - Implementation, metrics, confusion matrix
- ✅ (e) Support Vector Machine - Grid search over kernel and C parameters, metrics, confusion matrix
- ✅ (f) Random Forest - Implementation with parameter explanations, metrics, confusion matrix
- ✅ (g) Model Performance Comparison - Bar chart (accuracy), scatterplot (sensitivity vs specificity), analysis

**Current Work Location:** `Assign8.ipynb`

**Recent Commits:**
- `2a58cb2` - Problem 1(g) completed: model performance visualization and analysis
- `f15b7c6` - Problem 1(f) completed: Random Forest classifier
- `f889092` - Problem 1(e) completed: SVM with grid search
- `52f6e1b` - Problem 1(d) completed: Logistic Regression
- `1fe84f4` - Problem 1(c) completed: KNN with parameter tuning

### In Progress 🔄

**Problem 2: Neural Network Classifier**
- Pending implementation
- Target: Test accuracy ≥ 0.98
- Custom preprocessing and feature engineering required
- Will include updated comparison charts with Problem 1 models

---

## Dataset

**File:** `data/Spam_SMS.csv`

**Source:** SMS Spam Collection (public dataset for mobile spam research)

**Structure:**
- **Class:** Binary labels ('spam' or 'ham')
- **Message:** SMS text content

**Size:** 486,528 bytes

---

## Problems

### Problem 1: Traditional ML Classifiers

Build and compare multiple classification algorithms for SMS spam detection.

#### (a) Data Preparation
- Load dataset with pandas `read_csv()`
- Convert labels: 'spam' → 1, 'ham' → 0
- 80/20 train/test split (`random_state=42`)
- TF-IDF vectorization (exclude stop words)

#### (b) Naive Bayes Classifier
- Model: `MultinomialNB` (default parameters)
- Metrics: accuracy, sensitivity, specificity
- Visualization: confusion matrix

#### (c) k-Nearest Neighbors (KNN)
- Model: `KNeighborsClassifier`
- Parameter tuning: `n_neighbors` (explain default, tune, select best)
- Metrics: accuracy, sensitivity, specificity (best model)
- Visualization: confusion matrix

#### (d) Logistic Regression
- Model: `LogisticRegression` (default parameters)
- Metrics: accuracy, sensitivity, specificity
- Visualization: confusion matrix

#### (e) Support Vector Machine (SVM)
- Model: `SVC` (default, then tuned)
- Parameter analysis:
  - `kernel`: Explain default, effect of 'linear' → 'poly' on decision boundary
  - `C`: Explain default, effect on margin and support vectors
- Tuning: Grid search over `kernel` and `C` combinations
- Metrics: accuracy, sensitivity, specificity (best model)
- Visualization: confusion matrix

#### (f) Random Forest
- Model: `RandomForestClassifier` (default parameters)
- Parameter explanations:
  - `n_estimators`: Default value, impact on performance and variance
  - `max_depth`: Meaning of `max_depth=None`
  - `max_features`: Effect on tree diversity
- Metrics: accuracy, sensitivity, specificity
- Visualization: confusion matrix

#### (g) Model Performance Comparison
- **Bar Chart:** Test accuracy (sorted descending) → identify best model
- **Scatterplot:** Sensitivity vs. Specificity for all models
- **Analysis:** Discuss sensitivity/specificity trade-offs

---

### Problem 2: Neural Network Classifier

Build a neural network for SMS spam classification with custom preprocessing.

#### Requirements
- Preprocessing: Any method suitable for neural networks
- Feature engineering: Custom approach (not restricted to TF-IDF)
- Training & evaluation on same 80/20 split
- Metrics: test accuracy, sensitivity, specificity

#### Performance Target
**Test accuracy ≥ 0.98**

#### Comparison
- Recreate Problem 1 charts with neural network included
- Discuss performance differences and potential reasons

---

## Technical Stack

### Required Libraries
```python
# Data handling
pandas
numpy

# Machine Learning
scikit-learn
  - TfidfVectorizer
  - train_test_split
  - MultinomialNB
  - KNeighborsClassifier
  - LogisticRegression
  - SVC
  - RandomForestClassifier
  - confusion_matrix, accuracy_score

# Deep Learning (Problem 2)
tensorflow / keras  # or PyTorch

# Visualization
matplotlib
seaborn
```

---

## Key Concepts

### Model Parameters to Understand
1. **KNN `n_neighbors`** - Number of nearest neighbors to consider
2. **SVM `kernel`** - Kernel function for decision boundary (linear, poly, rbf)
3. **SVM `C`** - Regularization parameter (margin vs. misclassification trade-off)
4. **Random Forest `n_estimators`** - Number of trees in the forest
5. **Random Forest `max_depth`** - Maximum tree depth (None = unlimited)
6. **Random Forest `max_features`** - Features considered for splits (affects diversity)

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Sensitivity (Recall):** True positive rate
- **Specificity:** True negative rate
- **Confusion Matrix:** 2×2 performance breakdown

### Performance Trade-offs
- Sensitivity vs. Specificity balance
- Model complexity vs. generalization
- Training time vs. accuracy
- Interpretability vs. performance

---

## Deliverables

### Problem 1
- Implemented classifiers with proper data preparation
- Parameter explanations for KNN, SVM, Random Forest
- Performance metrics for all models
- Confusion matrices for all models
- Bar chart: sorted test accuracy
- Scatterplot: sensitivity vs. specificity
- Best model selection with justification

### Problem 2
- Neural network implementation
- Test accuracy ≥ 0.98
- Performance metrics
- Updated comparison charts
- Analysis of performance differences

---

## File Organization

```
8/
├── README.md                          # This file
├── Assign8.ipynb                      # Main implementation notebook
├── problem-requirements.txt          # Problem requirements
└── data/
    └── Spam_SMS.csv                  # Dataset (486KB)
```

---

## Getting Started

1. **Load dataset** from `data/Spam_SMS.csv`
2. **Follow Problem 1 workflow** systematically (a → g)
3. **Experiment with Problem 2** architecture for ≥0.98 accuracy
4. **Compare results** and analyze trade-offs

---

## Notes

- Use `random_state=42` for reproducibility
- TF-IDF should exclude English stop words
- Fit vectorizer on training data only, then transform both sets
- Grid search parameters systematically for SVM
- Neural network requires creative preprocessing for high accuracy

---

**Focus:** Text Classification & Machine Learning Comparison
