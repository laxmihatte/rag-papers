"""Build the vector index from PDFs in papers/.

Usage:
    python ingest.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.core import chunk_text, embed, read_pdf

PAPERS_DIR = Path("papers")
INDEX_DIR = Path("index")


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}/")

    all_chunks: list[str] = []
    metadata: list[dict] = []

    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        text = read_pdf(pdf)
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"source": pdf.name, "chunk": i})

    print(f"Embedding {len(all_chunks)} chunks (this may take a minute) ...")
    vectors = embed(all_chunks)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump({"chunks": all_chunks, "metadata": metadata}, f)

    print(f"Saved index: {vectors.shape[0]} vectors of dim {vectors.shape[1]} -> {INDEX_DIR}/")


if __name__ == "__main__":
    main()
