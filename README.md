# rag-papers

A minimal, fully-local **RAG** (retrieval-augmented generation) over a set of ML papers.
No API keys — everything runs on [Ollama](https://ollama.com).

- **Embeddings:** `nomic-embed-text`
- **Generation:** `llama3.2`
- **Vector store:** a plain NumPy array (cosine similarity), no database needed.

## Setup

```bash
# 1. Install Ollama and pull the models
ollama pull nomic-embed-text
ollama pull llama3.2

# 2. Python deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
# Build the index from PDFs in papers/ (writes to index/)
python ingest.py

# Ask questions
python ask.py "What is multi-head attention?"
python ask.py            # interactive
```

## How it works

1. `ingest.py` reads each PDF (`pypdf`), splits it into overlapping word chunks,
   embeds them, and saves unit-normalized vectors + chunk text to `index/`.
2. `ask.py` embeds your question, ranks chunks by cosine similarity, and feeds the
   top matches to the LLM as grounding context with source citations.

## Layout

```
papers/        source PDFs
rag/core.py    PDF reading, chunking, embedding
ingest.py      build the index
ask.py         query the index
index/          generated vector store (git-ignored)
```
