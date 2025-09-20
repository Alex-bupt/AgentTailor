# AgentTailor: Multi-Agent Long‑Tail Recommendation Framework

## Overview

AgentTailor reframes long‑tail recommendation as a multi‑agent reasoning pipeline, leveraging LLMs to profile users, analyze candidate items, and perform debiasing to improve exposure of niche content without sacrificing relevance.

## Repository Structure

```
├── few-shot.py      # Few-shot baseline: single-call LLM reranker with examples
├── zero-shot.py     # Zero-shot baseline: single-call LLM reranker without examples
├── rec_movie.py     # Full multi-agent pipeline for movie recommendation
├── rec_book.py      # Full multi-agent pipeline for book recommendation
├── requirements.txt # Python dependencies
└── README.md        # Project overview and usage instructions
```

## Requirements

- Python 3.8+
- Packages (see `requirements.txt`):
  - pandas
  - langchain
  - tqdm
  - concurrent‑futures
  - openai (via Deepseek API wrapper)

## Configuration

1. Set environment variables:
   
   ```bash
   export OPENAI_API_BASE="https://api.deepseek.com/v1"
   export OPENAI_API_KEY="<your-api-key>"
   ```

2. Place dataset files in expected directories (e.g., `ml-1m new/`, `bookcrossing new/`). Modify file paths in scripts if needed.

## Usage

### 1. Few‑Shot Baseline (`few-shot.py`)

- Purpose: Demonstrate a simple LLM-based reranker with two few-shot examples.

- Run:
  
  ```bash
  python few-shot.py
  ```

- Output: Prints Hit\@10 and NDCG\@10 metrics over a sample of users.

### 2. Zero‑Shot Baseline (`zero-shot.py`)

- Purpose: Evaluate the performance of an LLM reranker without examples.

- Run:
  
  ```bash
  python zero-shot.py
  ```

- Output: Evaluation metrics similar to few‑shot.

### 3. Movie Recommendation Pipeline (`rec_movie.py`)

- Purpose: Full AgentTailor pipeline for movies:
  
  1. **Agent 1**: Generate structured user profile.
  2. **Agent 2**: Analyze compatibility of candidates.
  3. **Agent 3**: Ranking based on analysis.
  4. **Agent 4**: Guided causal debiasing with λ parameter.

- Run:
  
  ```bash
  python rec_movie.py
  ```

- Output: Final evaluation report with Hits\@K, NDCG, rerank and debiasing failure counts.

### 4. Book Recommendation Pipeline (`rec_book.py`)

- Purpose: Adaptation of AgentTailor for book recommendations (BookCrossing dataset).

- Agents and steps mirror the movie pipeline, with ISBN-based compatibility analysis.

- Run:
  
  ```bash
  python rec_book.py
  ```

- Output: Evaluation metrics and detailed reranking logs.
