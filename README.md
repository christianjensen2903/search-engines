# Information Retrieval System: BM25, LMD & Query Expansion

A comprehensive implementation and evaluation of lexical retrieval systems on MS MARCO, exploring indexing strategies, ranking models, and query expansion techniques.

## 📋 Overview

This repository contains the complete implementation for experiments on:
- **Indexing strategies**: Stopword removal, stemming, and document length filtering
- **Ranking models**: BM25 and Language Modeling with Dirichlet smoothing (LMD)
- **Query expansion**: RM3 pseudo-relevance feedback, word embeddings, and LLM-based expansion
- **Rigorous evaluation**: Statistical significance testing with Holm-Bonferroni correction

All experiments use PyTerrier on a 200K-document MS MARCO subset with 4,434 queries.

## 🎯 Key Findings

### Best Configurations
| Model | Configuration | nDCG@10 | MRR@10 | Response Time |
|-------|--------------|---------|---------|---------------|
| **BM25** | Stopwords removed, k₁=3.5, b=0.75 | 0.586 | 0.486 | 47.6ms |
| **LMD** | Stopwords + stemming, μ=100 | 0.572 | 0.469 | 54.2ms |

### Impact of Preprocessing
- **Stopword removal**: ~9× faster search, ~13% smaller index
- **Stemming**: Modest gains in recall, slight increase in query time
- **Combined**: ~23% smaller index, ~7× faster retrieval

### Query Expansion Results
- **LLM-based (Flan-T5)**: Marginal improvements on BM25 (+0.002 MRR@10)
- **RM3**: No significant improvement over baselines
- **Word embeddings**: Consistently underperformed baselines

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/christianjensen2903/search-engines.git
cd search-engines

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the following files in the `data/` directory:

**Required files:**
- `docs.jsonl` - Document collection (title, url, body, docno)
- `train_queries.csv` - Training queries (tab-separated: qid, text)
- `train_qrels.csv` - Relevance judgments (tab-separated: qid, docno, relevance)
- `unseen_queries.csv` - Test queries

**Format examples:**

```json
// docs.jsonl
{"title": "Document title", "url": "...", "body": "content", "docno": "D123"}
```

```csv
# train_queries.csv
qid	text
1	what is information retrieval

# train_qrels.csv
qid	docno	relevance
1	D123	1
```

### Building Indexes

```python
# Open indexing.ipynb and run cells to build:
# - Full index
# - Stopwords removed
# - Stemming only
# - Stopwords + stemming
# (Each with/without length filtering)
```

Indexes are stored in `indexes/` directory.

## 📊 Experiments

### Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `indexing.ipynb` | Build all index variants |
| `data_analysis.ipynb` | Dataset statistics and visualizations |
| `seen_queries.ipynb` | BM25/LMD parameter tuning on train/validation |
| `query_expansion.ipynb` | RM3, word embeddings, LLM experiments |
| `unseen_queries.ipynb` | Final evaluation on held-out queries |
| `rm3_plots.ipynb` | RM3 parameter sensitivity analysis |
| `eval.ipynb` | Performance comparison and significance testing |

### Running Experiments

```python
# 1. Index building
jupyter notebook indexing.ipynb

# 2. Baseline retrieval
jupyter notebook seen_queries.ipynb

# 3. Query expansion
jupyter notebook query_expansion.ipynb

# 4. Final evaluation
jupyter notebook unseen_queries.ipynb
```

## 🔬 Methodology

### Parameter Tuning
- **BM25**: Grid search over k₁ ∈ [1.0, 5.0], b ∈ [0.0, 1.0]
- **LMD**: μ ∈ [50, 3000] (optimized to μ=100 for stopword-removed docs)
- **Optimization metric**: MRR (official MS MARCO metric)

### Evaluation Metrics
- nDCG, MRR, Precision, Recall @ {5, 10, 20}
- 95% confidence intervals computed via Student's t-distribution
- Statistical testing: One-sided paired t-tests with Holm-Bonferroni correction (α=0.05)

### Train/Validation Split
- Training: 3,547 queries (80%)
- Validation: 887 queries (20%)
- Unseen: 5,000 queries (held-out test set)
- Random seed: 42 (for reproducibility)

## 📈 Detailed Results

### Validation Set Performance

**BM25 (Stopwords Removed, k₁=3.5, b=0.75):**
```
nDCG@10: 0.586 ± 0.021
MRR@10:  0.486 ± 0.024
P@10:    0.091 ± 0.002
R@10:    0.906 ± 0.019
```

**LMD (Stopwords + Stemming, μ=100):**
```
nDCG@10: 0.572 ± 0.021
MRR@10:  0.469 ± 0.024
P@10:    0.090 ± 0.002
R@10:    0.903 ± 0.020
```

### Query Expansion Impact (Training Set)

| Method | nDCG@10 | MRR@10 | Response Time |
|--------|---------|---------|---------------|
| BM25 baseline | 0.581 | 0.484 | 20.7ms |
| BM25 + LLM (Q2E/ZS) | 0.581 | 0.483 | 110.5ms |
| BM25 + RM3 | 0.575 | 0.479 | 283.2ms |
| LMD baseline | **0.584** | **0.486** | 22.3ms |

## 🏗️ Repository Structure

```
search-engines/
├── data/                    # Data files (not included)
├── indexes/                 # Built indexes (generated)
├── indexing.ipynb          # Index construction
├── data_analysis.ipynb     # Dataset statistics
├── seen_queries.ipynb      # BM25/LMD experiments
├── query_expansion.ipynb   # QE experiments
├── unseen_queries.ipynb    # Final evaluation
├── rm3_plots.ipynb         # RM3 visualization
├── llm.ipynb               # LLM-based expansion
├── eval.ipynb              # Results analysis
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔧 Implementation Details

### Core Technologies
- **PyTerrier**: Indexing and retrieval framework
- **Flan-T5-Small**: LLM for query expansion (60M parameters)
- **GloVe**: Word embeddings (glove-wiki-gigaword-50)
- **Hardware**: Apple M1 (8-core CPU, 16GB RAM)

### Key Features
- Memory-efficient indexing (documents preloaded to RAM)
- No positional indexing (term positions not required)
- Batch processing for LLM inference
- Comprehensive logging and reproducibility

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{jensen2024searchengines,
  author = {Christian Mølholt Jensen},
  title = {Information Retrieval System: Evaluation of BM25, Language Modeling, and Query Expansion Techniques},
  school = {University of Copenhagen},
  year = {2024}
}
```

## 🔍 Key Insights

1. **Preprocessing matters**: Stopword removal provides greater gains than model selection
2. **BM25 excels at top ranks**: Superior MRR and early precision
3. **LMD better for recall**: Stronger performance at deeper cutoffs
4. **Query expansion is tricky**: Only LLM-based methods showed marginal gains; RM3 and embeddings hurt performance
5. **Low μ optimal**: μ=100 << avg doc length (1667 words), explained by skewed length distribution
6. **Generalization holds**: Validation and unseen performance closely track training metrics

## ⚠️ Limitations

- **Binary relevance labels**: MS MARCO provides only one relevant document per query
- **Sparse judgments**: Many documents lack relevance assessments
- **Query quality variation**: 8.7% of unseen queries achieved MRR@20 = 0 (vs. 2.6% for seen)
- **Latency**: LLM-based expansion adds ~100ms overhead (mitigable with GPU/batching)

## 🛣️ Future Work

- Multi-stage retrieval architectures (sparse + dense)
- Cross-encoder reranking
- Hard negative mining for query expansion
- Larger LLMs (e.g., Flan-T5-Large) with GPU acceleration
- Graded relevance judgments for better nDCG interpretation

---

**Note**: This work was completed as part of the UCPH course Search Engines. The complete report with detailed methodology, statistical analysis, and error analysis is available [here](https://github.com/christianjensen2903/search-engines/blob/main/Search_Engine_Course.pdf).
