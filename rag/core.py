"""Shared building blocks for the RAG pipeline.

Embeddings run locally via sentence-transformers (no API key, no server).
Generation runs on Groq's free hosted API (set GROQ_API_KEY).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from pypdf import PdfReader

# --- Models -----------------------------------------------------------------
# A small, fast embedding model that downloads once (~90 MB) and runs on CPU.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Groq-hosted model used to write the answer.
CHAT_MODEL = "llama-3.1-8b-instant"

# Chunking in words. Small overlap keeps sentences that straddle a boundary intact.
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50

PAPERS_DIR = Path("papers")
INDEX_DIR = Path("index")
TOP_K = 4


# --- PDF -> chunks ----------------------------------------------------------
def read_pdf(path: Path) -> str:
    """Extract all text from a PDF as a single string."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


# --- Embeddings -------------------------------------------------------------
@lru_cache(maxsize=1)
def _embedder():
    """Load the embedding model once and reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into an (n, dim) float32 array of unit vectors.

    normalize_embeddings=True makes every vector length 1, so a dot product
    equals cosine similarity.
    """
    vectors = _embedder().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.astype(np.float32)


# --- Index (build / load / search) ------------------------------------------
def build_index(papers_dir: Path = PAPERS_DIR, index_dir: Path = INDEX_DIR):
    """Read every PDF, chunk + embed it, and save the vector index to disk."""
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {papers_dir}/")

    chunks: list[str] = []
    metadata: list[dict] = []
    for pdf in pdfs:
        for i, chunk in enumerate(chunk_text(read_pdf(pdf))):
            chunks.append(chunk)
            metadata.append({"source": pdf.name, "chunk": i})

    vectors = embed(chunks)
    index_dir.mkdir(exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "chunks.json").write_text(json.dumps({"chunks": chunks, "metadata": metadata}))
    return vectors, chunks, metadata


def load_index(index_dir: Path = INDEX_DIR):
    """Load a previously built index from disk."""
    vectors = np.load(index_dir / "vectors.npy")
    data = json.loads((index_dir / "chunks.json").read_text())
    return vectors, data["chunks"], data["metadata"]


def ensure_index(papers_dir: Path = PAPERS_DIR, index_dir: Path = INDEX_DIR):
    """Load the index if it exists, otherwise build it. Used at app startup."""
    if (index_dir / "vectors.npy").exists():
        return load_index(index_dir)
    return build_index(papers_dir, index_dir)


def retrieve(question: str, vectors, chunks, metadata, k: int = TOP_K):
    """Return the k chunks most similar to the question."""
    q = embed([question])[0]
    scores = vectors @ q  # cosine similarity (vectors are unit-normalized)
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], metadata[i], float(scores[i])) for i in top]


# --- Prompt + generation ----------------------------------------------------
def build_prompt(question: str, hits) -> str:
    """Assemble the grounded prompt from retrieved chunks."""
    context = "\n\n".join(
        f"[{i + 1}] (from {meta['source']})\n{chunk}"
        for i, (chunk, meta, _) in enumerate(hits)
    )
    return (
        "Answer the question using only the context below. "
        "Cite sources with their [number]. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


@lru_cache(maxsize=1)
def _groq():
    from groq import Groq

    return Groq()  # reads GROQ_API_KEY from the environment


def generate_stream(prompt: str):
    """Stream the model's answer token by token from Groq."""
    stream = _groq().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
