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

### Project 2: Sequence Models — Forecasting and Sentiment
**Topics:** GRU time-series forecasting vs persistence baseline, IMDB sentiment, vocabulary-size effects, Embedding+LSTM vs bag-of-words
**Key Files:**
- `projects/2/README.md` - Project write-up with results
- `projects/2/imdb_common.py` - Shared IMDB data/model module
- `projects/2/part-1/`, `projects/2/part-2/`, `projects/2/part-3/` - Experiments (GRU forecasting, 200-word vocabulary study, LSTM)

### Project 3: Text Processing Fundamentals
**Topics:** NLTK vs spaCy pipeline comparison, bag-of-words with OOV handling, CNN baseline vs regularized variant on MNIST
**Key Files:**
- `projects/3/README.md` - Project write-up with results
- `projects/3/part-1/`, `projects/3/part-2/`, `projects/3/part-3/` - Experiments (library comparison, BoW, CNN study)

### Project 4: TF-IDF Representations & Word Embeddings
**Topics:** TF-IDF vocabulary transfer and OOV behavior, IMDB sentiment with TF-IDF n-grams (90% test-accuracy target), word2vec on Shakespeare (CBOW vs skip-gram)
**Key Files:**
- `projects/4/README.md` - Project write-up with results
- `projects/4/part-1/`, `projects/4/part-2/`, `projects/4/part-3/` - Experiments (vocabulary transfer, n-gram sentiment, word embeddings)

### Project 5: Character Embeddings & Autoencoders
**Topics:** Character-embedding transfer and OOV behavior, MNIST undercomplete autoencoder with a classifier-judged bottleneck, IMDb bag-of-words autoencoder scored against a frequency baseline
**Key Files:**
- `projects/5/README.md` - Project write-up with results
- `projects/5/part-1/`, `projects/5/part-2/`, `projects/5/part-3/` - Experiments (character embeddings, MNIST autoencoder, IMDb autoencoder)

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
