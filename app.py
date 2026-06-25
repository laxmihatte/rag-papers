"""Web UI for asking questions against the papers — a Gradio app.

Runs locally with `python app.py` and is the entry point Hugging Face Spaces
launches automatically. The index is built on first startup if missing.
"""

from __future__ import annotations

import gradio as gr

from rag.core import TOP_K, build_prompt, ensure_index, generate_stream, retrieve

# Build (or load) the index once when the app starts.
VECTORS, CHUNKS, METADATA = ensure_index()

EXAMPLES = [
    "What is multi-head attention?",
    "What is BERT pre-trained on?",
    "What is chain-of-thought prompting?",
]


def respond(question: str):
    """Stream the sources, then the answer, back to the UI."""
    question = (question or "").strip()
    if not question:
        yield "", ""
        return

    hits = retrieve(question, VECTORS, CHUNKS, METADATA, TOP_K)
    sources = "\n".join(
        f"[{i + 1}] **{meta['source']}** (score {score:.2f})"
        for i, (_, meta, score) in enumerate(hits)
    )

    answer = ""
    for token in generate_stream(build_prompt(question, hits)):
        answer += token
        yield sources, answer


with gr.Blocks(title="rag-papers") as demo:
    gr.Markdown("# 📄 Ask the papers\nRetrieval-augmented Q&A over a set of ML papers.")
    question = gr.Textbox(label="Question", placeholder="e.g. What is multi-head attention?")
    ask_btn = gr.Button("Ask", variant="primary")
    gr.Examples(EXAMPLES, inputs=question)
    sources = gr.Markdown(label="Sources")
    answer = gr.Markdown(label="Answer")

    ask_btn.click(respond, inputs=question, outputs=[sources, answer])
    question.submit(respond, inputs=question, outputs=[sources, answer])

if __name__ == "__main__":
    demo.queue().launch()
