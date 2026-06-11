# Problem 1: Topic Modeling with Latent Dirichlet Allocation (LDA)

## Overview

This solution implements topic modeling using Latent Dirichlet Allocation (LDA) on a research paper about Network-Knowledge Graph Duality in supply chain risk analysis. The paper sections are treated as separate documents for topic discovery.

## Assignment Requirements

The assignment required:

### (a) Article Selection and Topic Estimation
- **Requirement**: Select a comprehensive news article and manually estimate the number of topics
- **Implementation**: Used research paper "Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis" with 11 sections
- **Manual Topic Estimation**: **6 topics** identified:
  1. Large Language Models & Agent Architectures
  2. Knowledge Graphs & Network Science
  3. Supply Chain Risk & Dependencies
  4. Retrieval Systems & Data Integration
  5. Financial Applications & Factor Analysis
  6. System Implementation & Tools

### (b) Data Preparation
- **Requirement**: Split article into documents using paragraphs as delimiters, preprocess text
- **Implementation**:
  - Used the 11 sections from `article-sections.json` as separate documents
  - Preprocessing pipeline:
    - Convert to lowercase
    - Remove LaTeX artifacts and special characters
    - Remove stopwords using spaCy
    - Lemmatize words using spaCy's en_core_web_sm model
    - Filter: only alphabetic tokens with length > 2
  - Display preprocessing samples for verification

### (c) LDA Model Implementation
- **Requirement**: Implement LDA using gensim library
- **Implementation**:
  - Used `gensim.models.LdaModel`
  - Dictionary filtering: remove words appearing in <1 or >70% of documents
  - Model parameters:
    - `num_topics=6` (based on manual estimation)
    - `passes=20` (20 iterations over corpus)
    - `random_state=42` (reproducibility)
    - `alpha='auto'` and `eta='auto'` (automatic hyperparameter tuning)

### (d) Results Presentation
- **Requirement**: Present top 10 words per topic, top 2 associated documents per topic, and topic labels
- **Implementation**:
  - Displayed top 10 words with probabilities for each topic
  - Identified top 2 most associated sections for each topic
  - Generated concise 2-3 word labels based on top words and content
  - Created summary table with all results

## Project Structure

```
6/problem1/
├── README.md                    # This file
├── lda_topic_modeling.py        # Main implementation
├── article-sections.json        # Input: research paper sections
├── prd.txt                      # Problem requirements
├── venv/                        # Python 3.12 virtual environment
└── requirements.txt             # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.12 (required for gensim compatibility)
- pip (Python package manager)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   cd 6/problem1
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install gensim nltk spacy pandas matplotlib seaborn numpy
   ```

3. **Download spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Solution

Simply execute the main script:

```bash
python lda_topic_modeling.py
```

The script will:
1. Load NLTK data and spaCy model
2. Load and display the article sections
3. Show manual topic estimation
4. Preprocess all sections
5. Train the LDA model
6. Display comprehensive results including:
   - Top 10 words per topic
   - Topic labels
   - Top 2 most associated sections per topic
   - Summary table

## Results Summary

The LDA model successfully identified 6 distinct topics from the 11 paper sections:

| Topic ID | Label                   | Top 5 Words                                    | Strongest Section                       |
|----------|-------------------------|------------------------------------------------|-----------------------------------------|
| 0        | Knowledge Graphs        | product, graph, weight, input, location        | Conclusion and Next Steps               |
| 1        | LLM Applications        | graph, network, llm, chain, supply             | Introduction                            |
| 2        | Agent Systems           | tool, agent, llm, datum, user                  | System Overview                         |
| 3        | LLM Applications        | graph, model, time, traversal, query           | Background and Related Work             |
| 4        | Knowledge Graphs        | node, network, supply, coltan, apple           | Network-Science Path Discovery          |
| 5        | Financial Analysis      | factor, portfolio, context, shell, security    | Data Modes And Tools                    |

## Key Findings

- **Documents Analyzed**: 11 paper sections
- **Topics Identified**: 6 distinct topics
- **Dictionary Size**: 823 unique terms (after filtering)
- **Average Document Length**: 207.8 tokens
- **Total Corpus Terms**: 2,257 terms

The topics align well with the manual estimation, covering:
- LLM and agent architectures
- Knowledge graphs and network science
- Supply chain analysis (with concrete examples like Apple/coltan/DRC)
- Financial factor analysis and portfolio risk
- System implementation and data integration

## Technical Details

### Text Preprocessing
- **Tokenization**: spaCy tokenizer with lemmatization
- **Stopword Removal**: spaCy English stopwords
- **Additional Filtering**:
  - Removed LaTeX mathematical expressions
  - Removed reference markers and citations
  - Removed punctuation and special characters
  - Kept only alphabetic tokens with length > 2

### LDA Model Configuration
- **Algorithm**: Latent Dirichlet Allocation (gensim implementation)
- **Number of Topics**: 6 (manually estimated)
- **Training Passes**: 20 iterations
- **Hyperparameters**:
  - Alpha (document-topic density): Auto-tuned
  - Eta (topic-word density): Auto-tuned
- **Dictionary Filtering**:
  - Minimum document frequency: 1
  - Maximum document frequency: 70%

### Topic Labeling Strategy
Labels are generated automatically based on:
1. Top 7 words in the topic
2. Most associated section header
3. Keyword pattern matching for domain-specific terms:
   - Agent-related: "agent", "triage", "tool", "call"
   - Graph-related: "graph", "network", "node", "path", "edge"
   - Supply chain: "supply", "chain", "product", "company"
   - Financial: "factor", "portfolio", "security", "stock"

## Dependencies

```
gensim==4.4.0         # LDA implementation
nltk==3.9.2           # Natural language toolkit
spacy==3.8.7          # NLP preprocessing
pandas==2.3.3         # Data manipulation
numpy==2.3.4          # Numerical operations
matplotlib==3.10.7    # Plotting (for potential visualizations)
seaborn==0.13.2       # Statistical visualizations
```

## Notes

- The research paper is about using LLM agents with knowledge graphs for supply chain risk analysis
- Sections serve as natural "paragraphs" for topic modeling purposes
- The paper contains technical content with LaTeX formatting, which is cleaned during preprocessing
- Topic association scores are very high (>0.99) indicating strong topic-document relationships
- Some topics may overlap (e.g., multiple topics relate to graphs/networks) reflecting the paper's integrated approach

## Author

Solution implemented for CSCI E-89b NLP course, Assignment 6, Problem 1.
