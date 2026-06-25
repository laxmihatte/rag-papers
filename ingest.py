"""Build the vector index from PDFs in papers/.

Usage:
    python ingest.py

(The web app builds the index automatically on first run, so this is optional.)
"""

from rag.core import build_index

if __name__ == "__main__":
    vectors, chunks, _ = build_index()
    print(f"Saved index: {vectors.shape[0]} chunks, dim {vectors.shape[1]}")
