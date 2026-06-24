"""Ask a question against the indexed papers.

Usage:
    python ask.py "What is multi-head attention?"
    python ask.py            # interactive prompt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import ollama

from rag.core import CHAT_MODEL, embed

INDEX_DIR = Path("index")
TOP_K = 4


def load_index() -> tuple[np.ndarray, list[str], list[dict]]:
    if not (INDEX_DIR / "vectors.npy").exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    vectors = np.load(INDEX_DIR / "vectors.npy")
    data = json.loads((INDEX_DIR / "chunks.json").read_text())
    return vectors, data["chunks"], data["metadata"]


def retrieve(question: str, vectors, chunks, metadata, k: int = TOP_K):
    q = embed([question])[0]
    scores = vectors @ q  # cosine similarity (vectors are unit-normalized)
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], metadata[i], float(scores[i])) for i in top]


def answer(question: str) -> None:
    vectors, chunks, metadata = load_index()
    hits = retrieve(question, vectors, chunks, metadata)

    context = "\n\n".join(
        f"[{i + 1}] (from {meta['source']})\n{chunk}"
        for i, (chunk, meta, _) in enumerate(hits)
    )
    prompt = (
        "Answer the question using only the context below. "
        "Cite sources with their [number]. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    print("\nRetrieved:")
    for i, (_, meta, score) in enumerate(hits):
        print(f"  [{i + 1}] {meta['source']} (chunk {meta['chunk']}, score {score:.3f})")
    print("\nAnswer:\n")

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print()


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    if question:
        answer(question)


if __name__ == "__main__":
    main()
