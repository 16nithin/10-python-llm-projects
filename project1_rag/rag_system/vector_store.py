"""
vector_store.py - Local hash-based embeddings (no API needed for indexing)
Gemini is still used for generating answers. Only the embedding step is local.
"""

import os
import json
import hashlib
import numpy as np
import faiss
from typing import List, Dict, Any


class VectorStore:
    EMBEDDING_DIM = 768

    def __init__(self, api_key, collection_name="rag_documents", persist_dir="./faiss_db"):
        self.api_key = api_key  # kept for compatibility
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, f"{collection_name}.bin")
        self.metadata_path = os.path.join(persist_dir, f"{collection_name}_metadata.json")
        self._load_or_create_index()
        print(f"✅ Vector store ready. Documents indexed: {len(self.metadata)}")

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("📂 Loading existing index from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            print("🆕 Creating new FAISS index...")
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.metadata = []

    def _save_to_disk(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _embed(self, texts):
        """
        Local word-frequency embedding using deterministic hashing.
        No API calls — works completely offline using only numpy.
        Each word is hashed to one of 768 dimensions and its frequency counted.
        Similar documents (sharing words) get similar vectors.
        BM25 handles exact keyword retrieval; this adds document-level similarity.
        """
        all_embeddings = []
        for text in texts:
            vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                # Deterministic hash — same word always maps to same dimension
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.EMBEDDING_DIM
                vec[h] += 1.0
            all_embeddings.append(vec)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms

    def add_documents(self, chunks):
        if not chunks:
            print("⚠️  No chunks to add.")
            return
        existing_ids = {m.get("chunk_id") for m in self.metadata}
        new_chunks = [c for c in chunks if c.get("chunk_id") not in existing_ids]
        if not new_chunks:
            print("ℹ️  All chunks already indexed.")
            return
        print(f"🔄 Indexing {len(new_chunks)} chunks locally...")
        texts = [chunk["text"] for chunk in new_chunks]
        embeddings = self._embed(texts)
        self.index.add(embeddings)
        for chunk in new_chunks:
            self.metadata.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk.get("chunk_id", ""),
                "start_char": chunk.get("start_char", 0),
                "end_char": chunk.get("end_char", 0)
            })
        self._save_to_disk()
        print(f"✅ Added {len(new_chunks)} chunks. Total: {len(self.metadata)}")

    def semantic_search(self, query, n_results=10):
        if not self.metadata:
            return []
        query_embedding = self._embed([query])
        n = min(n_results, len(self.metadata))
        scores, indices = self.index.search(query_embedding, n)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "score": float(score),
                "rank": rank + 1,
                "search_type": "semantic"
            })
        return results

    def get_all_documents(self):
        return [{"text": m["text"], "source": m["source"]} for m in self.metadata]

    def count(self):
        return len(self.metadata)

    def clear(self):
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.metadata = []
        self._save_to_disk()
        print("🗑️  Vector store cleared.")
