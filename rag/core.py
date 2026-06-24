"""Shared building blocks for the RAG pipeline: PDF reading, chunking, embedding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

# Local Ollama models (pulled via `ollama pull <name>`).
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

# Chunking in words. Small overlap keeps sentences that straddle a boundary intact.
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50


def read_pdf(path: Path) -> str:
    """Extract all text from a PDF as a single string."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


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


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into an (n, dim) float32 array of unit vectors."""
    vectors = []
    for text in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        vectors.append(resp["embedding"])
    arr = np.array(vectors, dtype=np.float32)
    # Normalize so a dot product equals cosine similarity.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
