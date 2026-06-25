---
title: RAG Papers
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# rag-papers

A minimal **RAG** (retrieval-augmented generation) over a set of ML papers, built to
deploy **free and always-on** on [Hugging Face Spaces](https://huggingface.co/spaces).

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) — runs in-process, no API key.
- **Generation:** `llama-3.1-8b-instant` on [Groq](https://groq.com)'s free API.
- **Vector store:** a plain NumPy array (cosine similarity), no database.

The block of `---` metadata at the top of this file is what Hugging Face reads to
configure the Space (Gradio app, entry point `app.py`).

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...        # free key from https://console.groq.com

python app.py                  # web UI at http://127.0.0.1:7860
# or
python ask.py "What is multi-head attention?"
```

The index is built automatically on first run (from the PDFs in `papers/`).

## Deploy free on Hugging Face Spaces

See [DEPLOY.md](DEPLOY.md). Short version: create a Gradio Space, push this repo to
it, and add `GROQ_API_KEY` as a Space secret. You get a public URL like
`https://<user>-rag-papers.hf.space`.

## How it works

1. Each PDF is read (`pypdf`), split into overlapping word chunks, and embedded into
   unit vectors saved in `index/`.
2. A question is embedded the same way; chunks are ranked by cosine similarity
   (`vectors @ q`); the top matches are handed to the LLM as cited context.

## Layout

```
papers/        source PDFs
rag/core.py    PDF reading, chunking, embedding, retrieval, generation
ingest.py      build the index (optional; the app does it on startup)
ask.py         command-line Q&A
app.py         Gradio web UI (Hugging Face Spaces entry point)
```
