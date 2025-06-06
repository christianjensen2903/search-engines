# Search Engines Project

This repository contains code and experiments for various search engine implementations and evaluations.

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

- `data/`: Directory for storing datasets
- `indexes/`: Directory for storing search engine indexes

## Setting Up Data

Place the following files in the `data/` directory:

1. `docs.jsonl`: A JSONL file containing the document collection with the following format:

   ```json
   {
     "title": "Document title",
     "url": "Document url",
     "body": "Document content",
     "docno": "Document id"
   }
   ```

2. `train_queries.csv`: A tab-separated file containing training queries with the following format:

   ```
   qid    text
   1      example query text
   ```

3. `train_qrels.csv`: A tab-separated file containing relevance judgments with the following format:

   ```
   qid    docno    relevance
   1      D123     1
   ```

4. `unseen_queries.csv`: A tab-separated file containing test queries with the following format:

   ```
   qid    text
   1      test query text
   ```

5. `runs_scores.csv`: A CSV file containing evaluation scores for different runs with the following format:

   ```
   nDCG@5,nDCG@10,nDCG@20,R@5,R@10,R@20,P@5,P@10,P@20,RR@5,RR@10,RR@20,MRT,run_name
   0.5158,0.5635,0.5785,0.7292,0.875,0.9335,0.1458,0.0875,0.0467,0.4442,0.4645,0.4686,1.1283,pei246PEI642
   ```

6. `run_scores_perquery.csv`: A CSV file containing per-query evaluation scores with the following format:
   ```
   qid,run_name,nDCG@5,nDCG@10,nDCG@20,R@5,R@10,R@20,P@5,P@10,P@20,RR@5,RR@10,RR@20
   1,pei246PEI642,0.5158,0.5635,0.5785,0.7292,0.875,0.9335,0.1458,0.0875,0.0467,0.4442,0.4645,0.4686,1.1283
   ```

## Building Indexes

Before running experiments, you need to build the necessary indexes:

1. Open `indexing.ipynb` in Jupyter Notebook
2. Follow the instructions in the notebook to build the required indexes
3. The built indexes will be stored in the `indexes/` directory

## Running Experiments

The project contains several Jupyter notebooks for different experiments:

- `query_expansion.ipynb`: Query expansion experiments
- `unseen_queries.ipynb`: Experiments with unseen queries
- `seen_queries.ipynb`: Experiments with seen queries
- `llm.ipynb`: LLM-based experiments
- `eval.ipynb`: Evaluation of search results
- `data_analysis.ipynb`: Data analysis and visualization
- `rm3_plots.ipynb`: RM3 algorithm visualization
