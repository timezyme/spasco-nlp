# NLP-ML projects

A collection of natural language processing implementations in Python, progressing from text-processing fundamentals to machine learning and deep learning applications.

---

## Environment Setup

### Python Virtual Environment
- **Location:** `venv/` (project root, create locally — gitignored)
- **Setup:**
  ```bash
  python3.12 -m venv venv   # 3.12: newest Python with TensorFlow wheel support
  source venv/bin/activate
  pip install nltk spacy scikit-learn tensorflow keras pandas matplotlib seaborn jupyter
  python -m spacy download en_core_web_sm
  ```

### Key Libraries
- **Deep Learning:** Keras 3.x (standalone), TensorFlow
- **NLP:** NLTK, spaCy
- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Jupyter:** notebook, nbconvert

**Important:** Always use `venv/bin/python` for running scripts to avoid environment conflicts.

---

## Projects

### Project 1: Reuters Topic Classification
**Topics:** Multi-class text classification (46 topics), epoch selection, architecture tuning, optimizer comparison, deep vs classical baselines
**Key Files:**
- `projects/1/README.md` - Project write-up with results
- `projects/1/reuters_common.py` - Shared data/model/metrics module
- `projects/1/part-1/`, `projects/1/part-2/`, `projects/1/part-3/` - Experiments (baseline, improved model, optimizer study)

### Project 2: Text Classification Basics
**Topics:** Movie review classification, sentiment analysis
**Key Files:**
- `projects/2/3.5-classifying-movie-reviews.ipynb` - Implementation
- `projects/2/README.md` - Lessons learned (environment setup)

### Project 3: Text Processing
**Topics:** Text preprocessing, bag-of-words, feature engineering

### Project 4: Model Evaluation
**Topics:** Model evaluation metrics, performance analysis
**Key Files:**
- `projects/4/3.5-classifying-movie-reviews.ipynb` - Implementation
- `projects/4/test_accuracy_90plus.txt` - Performance results

### Project 5: Character Embeddings & Autoencoders
**Topics:** Character-level embeddings, sequence modeling, autoencoders
**Key Files:**
- `projects/5/problem1/` - Character embedding solutions
- `projects/5/problem2/` - Text generation with RNNs
- `projects/5/problem3/` - Autoencoder for IMDb reviews
- `projects/5/PROBLEM3_SOLUTION_MEMORY.md` - Technical documentation

**Highlights:**
- Character-level embedding with unknown token handling
- LSTM-based character prediction models
- Autoencoder architecture for text representation

### Project 6: Topic Modeling
**Topics:** LDA topic modeling, deep neural network architectures
**Key Files:**
- `projects/6/problem-1-2-3/` - Problem implementations

### Project 7: R-based Analysis
**Topics:** R programming for NLP, statistical analysis
**Key Files:**
- `projects/7/Assign7.Rmd` - R Markdown source
- `projects/7/Assign7.html` - HTML report
- `projects/7/data/` - Analysis datasets

### Project 8: Text Classification with Machine Learning
**Topics:** SMS spam classification, ML algorithm comparison, neural networks
**Key Files:**
- `projects/8/Assign8.ipynb` - Main implementation notebook
- `projects/8/data/Spam_SMS.csv` - SMS spam dataset
- `projects/8/README.md` - Detailed problem requirements

**Highlights:**
- Naive Bayes, KNN, Logistic Regression, SVM, and Random Forest comparison with TF-IDF features
- Neural network classifier reaching ≥ 0.98 test accuracy

### Project 9: NER & Document Classification
**Topics:** Named entity recognition with spaCy, news classification
**Key Files:**
- `projects/9/Assign9.ipynb` - Main implementation notebook
- `projects/9/data/` - News dataset

### Final Project
**Topics:** Knowledge-graph-based question answering over research papers
**Key Files:**
- `final-project/final-project-draft.md` - Design draft

---

## Repository Structure

```
spasco-nlp/
├── README.md                     # This file
├── venv/                         # Python virtual environment (create locally, gitignored)
├── projects/                     # Numbered project directories (1-9)
│   ├── *.ipynb                   # Jupyter notebooks
│   ├── *.docx                    # Write-ups
│   └── README.md                 # Project-specific docs
├── final-project/                # Final project materials
└── guides/                       # Reference materials
```

---

## Development Workflow

1. **Setup**: Activate venv - `source venv/bin/activate`
2. **Branch**: Create feature branch - `git checkout -b problem-X`
3. **Implement**: Work in Jupyter notebook
4. **Test**: Validate results and metrics
5. **Document**: Update README with progress
6. **Commit**: Incremental commits with descriptive messages
7. **Merge**: Merge to main when complete

---

## Key Learnings & Best Practices

### Environment Management
- Always use the project venv (`venv/bin/python`)
- Verify the environment before importing libraries
- Check for existing venv directories before creating new ones
- Never use system Python or conda base for project work

### Import Patterns
- Keras 3.x: `from keras import models, layers`
- Avoid: `from tensorflow.keras import ...` (compatibility issues)

### Code Organization
- Jupyter notebooks for primary implementations
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
- **VS Code:** Primary editor

---

## Resources

Reference materials live in `guides/`:
- LLMs in the GitHub Ecosystem
- Software Architecture Patterns
- NLP-ML prompt engineering

Several project directories also include datasets in `data/`.
