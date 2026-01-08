import os, json, glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

DATA_RAW = "data/raw"
DATA_INDEX = "data/index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900        # chars
CHUNK_OVERLAP = 150

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append((i + 1, txt))
    return pages

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = " ".join(t.split())
    return t.strip()

def chunk_text(
    text: str,
    source: str,
    page: int = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        meta = {"source": source}
        if page is not None:
            meta["page"] = page
        chunks.append(Chunk(id=f"{source}:{page or 0}:{idx}", text=chunk, meta=meta))
        idx += 1
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def ingest_documents(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    os.makedirs(DATA_INDEX, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(DATA_RAW, "**/*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p) and (p.endswith(".txt") or p.endswith(".pdf") or p.endswith(".md"))]
    if not paths:
        raise SystemExit("No .txt/.md/.pdf files found in data/raw")

    all_chunks: List[Chunk] = []

    for p in paths:
        name = os.path.relpath(p, DATA_RAW)
        if p.endswith(".pdf"):
            for page_no, page_text in read_pdf(p):
                all_chunks.extend(chunk_text(page_text, source=name, page=page_no, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        else:
            all_chunks.extend(chunk_text(read_txt(p), source=name, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    all_chunks = [c for c in all_chunks if len(c.text) > 40]
    print(f"Chunks created: {len(all_chunks)}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in all_chunks]

    vectors = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        batch = texts[i:i+64]
        emb = model.encode(batch, normalize_embeddings=True)
        vectors.append(emb)
    X = np.vstack(vectors).astype("float32")

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalize + inner product
    index.add(X)

    faiss.write_index(index, os.path.join(DATA_INDEX, "faiss.index"))
    meta = [{"id": c.id, "text": c.text, "meta": c.meta} for c in all_chunks]
    with open(os.path.join(DATA_INDEX, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Index written to data/index (faiss.index + chunks.json)")
    return len(all_chunks)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    ingest_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

if __name__ == "__main__":
    main()
