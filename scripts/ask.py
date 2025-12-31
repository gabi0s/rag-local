import os, json
from typing import List, Dict, Any
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

DATA_INDEX = "data/index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
TOP_K = 5

def load_store():
    index = faiss.read_index(os.path.join(DATA_INDEX, "faiss.index"))
    with open(os.path.join(DATA_INDEX, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def embed_query(model, q: str) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True).astype("float32")
    return v

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        src = c["meta"].get("source", "unknown")
        page = c["meta"].get("page", None)
        ref = f"{src}" + (f" p.{page}" if page else "")
        ctx_blocks.append(f"[Source {i}: {ref}]\n{c['text']}")
    context_text = "\n\n".join(ctx_blocks)

    return f"""Tu es un assistant. Réponds en français.
Règles:
- Utilise UNIQUEMENT les informations dans les sources.
- Si l’info n’est pas dans les sources, dis clairement "Je ne sais pas d’après les documents fournis".
- À la fin, liste les sources utilisées.

SOURCES:
{context_text}

QUESTION:
{question}

RÉPONSE:
"""

def call_ollama(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]

def main():
    import sys
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/ask.py "ta question"')

    index, chunks = load_store()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    qv = embed_query(embedder, question)
    scores, ids = index.search(qv, TOP_K)

    contexts = []
    for idx in ids[0]:
        if idx == -1:
            continue
        contexts.append(chunks[int(idx)])

    prompt = build_prompt(question, contexts)
    answer = call_ollama(prompt)

    print("\n=== RÉPONSE ===\n")
    print(answer.strip())
    print("\n=== SOURCES RETROUVÉES ===\n")
    for i, c in enumerate(contexts, 1):
        src = c["meta"].get("source", "unknown")
        page = c["meta"].get("page", None)
        ref = f"{src}" + (f" p.{page}" if page else "")
        print(f"{i}. {ref}")

if __name__ == "__main__":
    main()
