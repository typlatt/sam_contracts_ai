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

### 4. Seed the Database
Before you can build the index or query contracts, you must seed the DuckDB database with contract data. Use the provided seeding script:
```bash
python Scripts/seed.py
```
This will load the relevant CSVs into `data/sam.duckdb`.

### 5. Build the Semantic Index
Run the indexing script to create the FAISS index and description CSV:
```bash
python Scripts/build_index.py
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
- `Scripts/` — Utility and setup scripts (seeding, indexing, etc.)
- `Scripts/utills.py` — Utility functions for embedding, batching, context retrieval, and more
- `query_rag.py` — Interactive script for querying the index
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
