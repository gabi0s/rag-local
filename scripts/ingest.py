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

CHUNK_SIZE = 900        # en caractères (simple)
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

def chunk_text(text: str, source: str, page: int = None) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunk = text[start:end]
        meta = {"source": source}
        if page is not None:
            meta["page"] = page
        chunks.append(Chunk(id=f"{source}:{page or 0}:{idx}", text=chunk, meta=meta))
        idx += 1
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def main():
    os.makedirs(DATA_INDEX, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(DATA_RAW, "**/*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p) and (p.endswith(".txt") or p.endswith(".pdf") or p.endswith(".md"))]
    if not paths:
        raise SystemExit("Aucun fichier .txt/.md/.pdf trouvé dans data/raw")

    all_chunks: List[Chunk] = []

    for p in paths:
        name = os.path.relpath(p, DATA_RAW)
        if p.endswith(".pdf"):
            for page_no, page_text in read_pdf(p):
                all_chunks.extend(chunk_text(page_text, source=name, page=page_no))
        else:
            all_chunks.extend(chunk_text(read_txt(p), source=name))

    all_chunks = [c for c in all_chunks if len(c.text) > 40]
    print(f"✅ Chunks créés: {len(all_chunks)}")

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

    print("Index écrit dans data/index (faiss.index + chunks.json)")

if __name__ == "__main__":
    main()
