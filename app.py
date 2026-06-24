"""Tiny web UI for asking questions against the indexed papers.

Usage:
    python app.py
    # then open http://127.0.0.1:5000
"""

from __future__ import annotations

from flask import Flask, Response, render_template_string, request

import ollama

from rag.core import CHAT_MODEL
from ask import TOP_K, load_index, retrieve

app = Flask(__name__)

# Load the index once at startup, not on every request.
VECTORS, CHUNKS, METADATA = load_index()

PAGE = """
<!doctype html>
<title>rag-papers</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 1.4rem; }
  textarea { width: 100%; padding: 10px; font-size: 1rem; box-sizing: border-box; }
  button { margin-top: 10px; padding: 8px 18px; font-size: 1rem; cursor: pointer; }
  #sources { color: #555; font-size: .85rem; margin: 16px 0; white-space: pre-line; }
  #answer { white-space: pre-wrap; line-height: 1.5; }
</style>
<h1>📄 Ask the papers</h1>
<textarea id="q" rows="3" placeholder="e.g. What is multi-head attention?"></textarea>
<button onclick="ask()">Ask</button>
<div id="sources"></div>
<div id="answer"></div>
<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const sources = document.getElementById('sources');
  const answer = document.getElementById('answer');
  sources.textContent = 'Retrieving...';
  answer.textContent = '';
  const resp = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: q})
  });
  // The response streams: first line is the sources header, the rest is the answer.
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let header = false;
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    let text = decoder.decode(value, {stream: true});
    if (!header) {
      const nl = text.indexOf('\\n\\n');
      sources.textContent = text.slice(0, nl);
      answer.textContent += text.slice(nl + 2);
      header = true;
    } else {
      answer.textContent += text;
    }
  }
}
</script>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/ask")
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return Response("No question.\n\n", mimetype="text/plain")

    hits = retrieve(question, VECTORS, CHUNKS, METADATA, k=TOP_K)
    context = "\n\n".join(
        f"[{i + 1}] (from {meta['source']})\n{chunk}"
        for i, (chunk, meta, _) in enumerate(hits)
    )
    prompt = (
        "Answer the question using only the context below. "
        "Cite sources with their [number]. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def generate():
        # First chunk: a human-readable "sources" header, then a blank-line separator.
        header = "Sources: " + ", ".join(
            f"[{i + 1}] {meta['source']} (score {score:.2f})"
            for i, (_, meta, score) in enumerate(hits)
        )
        yield header + "\n\n"
        # Then stream the model's answer token by token.
        for part in ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            yield part["message"]["content"]

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
