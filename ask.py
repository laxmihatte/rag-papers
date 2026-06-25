"""Ask a question against the indexed papers from the command line.

Usage:
    python ask.py "What is multi-head attention?"
    python ask.py            # interactive prompt
"""

from __future__ import annotations

import sys

from rag.core import (
    TOP_K,
    build_prompt,
    ensure_index,
    generate_stream,
    retrieve,
)


def answer(question: str) -> None:
    vectors, chunks, metadata = ensure_index()
    hits = retrieve(question, vectors, chunks, metadata, TOP_K)

    print("\nRetrieved:")
    for i, (_, meta, score) in enumerate(hits):
        print(f"  [{i + 1}] {meta['source']} (chunk {meta['chunk']}, score {score:.3f})")
    print("\nAnswer:\n")

    for token in generate_stream(build_prompt(question, hits)):
        print(token, end="", flush=True)
    print()


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    if question:
        answer(question)


if __name__ == "__main__":
    main()
