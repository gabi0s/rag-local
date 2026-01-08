import os
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ingest import ingest_documents, CHUNK_SIZE, CHUNK_OVERLAP

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_INDEX = os.path.join(BASE_DIR, "data", "index")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
TOP_K_DEFAULT = 5

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoreCache:
    def __init__(self) -> None:
        self.index = None
        self.chunks = None
        self.mtime = None

    def _store_mtime(self) -> Optional[float]:
        chunks_path = os.path.join(DATA_INDEX, "chunks.json")
        index_path = os.path.join(DATA_INDEX, "faiss.index")
        if not (os.path.exists(chunks_path) and os.path.exists(index_path)):
            return None
        return max(os.path.getmtime(chunks_path), os.path.getmtime(index_path))

    def load(self) -> None:
        mtime = self._store_mtime()
        if mtime is None:
            raise FileNotFoundError("Index not found. Run ingest first.")
        if self.mtime == mtime and self.index is not None and self.chunks is not None:
            return
        self.index = faiss.read_index(os.path.join(DATA_INDEX, "faiss.index"))
        with open(os.path.join(DATA_INDEX, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.mtime = mtime

    def invalidate(self) -> None:
        self.mtime = None

store = StoreCache()
embedder = None

def get_embedder() -> SentenceTransformer:
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return embedder

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
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

    return (
        "You are a helpful assistant. Answer in French.\n"
        "Rules:\n"
        "- Use only information from the sources.\n"
        "- If the answer is not in the sources, say you do not know.\n"
        "- At the end, list the sources used.\n\n"
        "SOURCES:\n"
        f"{context_text}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
    )

def retrieve_contexts(question: str, top_k: int) -> List[Dict[str, Any]]:
    store.load()
    model = get_embedder()
    qv = embed_query(model, question)
    scores, ids = store.index.search(qv, top_k)

    contexts = []
    for idx in ids[0]:
        if idx == -1:
            continue
        contexts.append(store.chunks[int(idx)])
    return contexts

def stream_ollama(prompt: str):
    with requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break

def sse_event(event: str, data: str) -> str:
    lines = str(data).splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"

class IngestRequest(BaseModel):
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/docs")
def list_docs():
    docs = []
    raw_map = {}
    if os.path.exists(DATA_RAW):
        for root, _, files in os.walk(DATA_RAW):
            for name in files:
                if not (name.endswith(".txt") or name.endswith(".pdf") or name.endswith(".md")):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, DATA_RAW).replace("\\", "/")
                doc = {
                    "name": rel,
                    "size": os.path.getsize(path),
                    "modified": int(os.path.getmtime(path)),
                    "chunks": None,
                }
                docs.append(doc)
                raw_map[rel] = doc

    chunks_path = os.path.join(DATA_INDEX, "chunks.json")
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        counts = {}
        for c in chunks:
            src = c.get("meta", {}).get("source")
            if not src:
                continue
            counts[src] = counts.get(src, 0) + 1
        for src, count in counts.items():
            if src in raw_map:
                raw_map[src]["chunks"] = count
            else:
                docs.append({
                    "name": src,
                    "size": None,
                    "modified": None,
                    "chunks": count,
                })

    docs.sort(key=lambda d: d["name"].lower())
    return {"docs": docs}

@app.post("/api/docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    if not os.path.exists(DATA_RAW):
        os.makedirs(DATA_RAW, exist_ok=True)
    saved = []
    for up in files:
        if not (up.filename.endswith(".txt") or up.filename.endswith(".pdf") or up.filename.endswith(".md")):
            raise HTTPException(status_code=400, detail=f"Unsupported file: {up.filename}")
        safe_name = os.path.basename(up.filename)
        target = os.path.join(DATA_RAW, safe_name)
        with open(target, "wb") as f:
            while True:
                chunk = await up.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        saved.append(safe_name)
    return {"saved": saved}

@app.post("/api/ingest")
def ingest(req: IngestRequest):
    count = ingest_documents(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    store.invalidate()
    return {"chunks": count}

@app.get("/api/chat/stream")
def chat_stream(question: str = Query(..., min_length=1), top_k: int = TOP_K_DEFAULT):
    try:
        contexts = retrieve_contexts(question, top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt = build_prompt(question, contexts)

    def event_stream():
        for token in stream_ollama(prompt):
            yield sse_event("token", token)
        sources = []
        for c in contexts:
            src = c["meta"].get("source", "unknown")
            page = c["meta"].get("page", None)
            sources.append({"source": src, "page": page})
        yield sse_event("sources", json.dumps(sources))
        yield sse_event("done", "1")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
