# Roadmap: from minimal RAG to research-paper assistant

The current app is a minimal RAG over 5 hand-picked PDFs. This roadmap grows it
into an assistant that ingests arXiv papers and supports Q&A, cross-paper
comparison, and (stretch) literature-review generation, all with verifiable
citations. The stack stays **free**: local embeddings and reranker, Groq for
generation, Hugging Face Spaces for hosting.

## Stack

| Piece | Choice | Why |
|---|---|---|
| Fetching | `arxiv` package | Search by query/ID, gives clean metadata |
| PDF parsing | `PyMuPDF` | Fast, exposes font sizes for detecting section headings |
| Embeddings | `all-MiniLM-L6-v2` (baseline) → `bge-small-en-v1.5` | Both run on CPU in a free Space |
| Keyword search | `rank_bm25` | Exact model/dataset/metric names |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) | Local, free |
| Generation | `llama-3.1-8b-instant` on Groq | Free API |
| UI | Gradio | Already deployed on HF Spaces |

## Milestones

### 1. arXiv ingestion
- [ ] `fetch.py`: download papers by arXiv ID list or search query into `papers/`
- [ ] Save metadata per paper (`papers/metadata.json`): arXiv ID + version, title,
      authors, year, categories, abstract
- [ ] Deduplicate versions (keep latest `vN`)
- [ ] Grow corpus to ~20 LLM papers

### 2. Section-aware chunking
- [ ] Parse with PyMuPDF; detect section headings
- [ ] Chunk within sections; attach `paper_id, title, section, page, year`
- [ ] Drop or tag the References section so it doesn't pollute retrieval
- [ ] Inspect parsed output by hand for 3 papers (two-column layouts, tables)

### 3. Hybrid search + reranker
- [ ] BM25 index next to the vector index
- [ ] Fuse scores (reciprocal rank fusion)
- [ ] Rerank top-20 → top-k with a cross-encoder

### 4. Evaluation
- [ ] `eval/questions.jsonl`: 30+ questions with known paper + section
- [ ] Recall@k script
- [ ] Ablation table: fixed-window vs section chunks, vector vs hybrid, ± reranker

### 5. Citations in Q&A
- [ ] Filter by `paper_id` when the question names a paper
- [ ] Cite as `[Author Year, §Section, p.N]`, built from metadata
- [ ] "Not found in the provided papers" instead of guessing
- [ ] Verify every inline citation maps to a retrieved chunk

### 6. Comparison mode
- [ ] Per-paper, per-aspect retrieval (method, datasets, metrics, results, limitations)
- [ ] Structured comparison table + short synthesis
- [ ] Gradio tab for picking papers

### 7. Stretch: literature review
- [ ] Per-paper structured notes → cluster → per-cluster sections
- [ ] Reference list generated from metadata, never by the LLM
