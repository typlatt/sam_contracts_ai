# SAM Contracts AI

A Retrieval-Augmented Generation (RAG) system for querying and analyzing US government contract data, using OpenAI embeddings and FAISS for semantic search.

---

## Features
- Build a semantic index of contract descriptions using OpenAI embeddings and FAISS
- Query the index with natural language questions
- Retrieve relevant contract summaries and generate answers using OpenAI's GPT models
- Support for custom personas and context-aware answers

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd sam_contracts_ai
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare environment variables
Create a `.env` file or set the following in your shell:
```
OPENAI_API_KEY=your_openai_api_key
```

### 4. Data Preparation
Ensure the following data files are present in the `data/` directory:
- `sam_contract_ops_071225.csv` (main contracts CSV)
- `sam_contracts_archived_2024.csv` (archived contracts)

### 5. Build the Semantic Index
Run the indexing script to create the FAISS index and description CSV:
```bash
python build_index.py
```

### 6. Query the Index
Start the interactive Q&A loop:
```bash
python query_rag.py
```

You will be prompted to enter natural language questions about the contracts.

---

## Example Questions
- What are the recent contracts related to drone technology?
- What are contracts related to data or analytics?
- Which contracts are relevant for small businesses?
- List contracts awarded in the last year for cybersecurity.

---

## Project Structure
- `build_index.py` — Script to build the FAISS index from contract descriptions
- `query_rag.py` — Interactive script for querying the index
- `utills.py` — Utility functions for embedding, batching, context retrieval, and more
- `constants.py` — File paths and persona definitions
- `data/` — Directory for CSVs, FAISS index, and DuckDB database

---

## Requirements
- Python 3.9+
- Packages: faiss, numpy, pandas, duckdb, openai, tiktoken
- OpenAI API key

---

## Notes
- Large data files may require sufficient disk space and RAM.
- For local LLM support, see `ollama.py`.

---

## License
