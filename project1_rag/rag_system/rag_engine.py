"""
rag_engine.py - RAG pipeline using Gemini REST API directly
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from document_processor import process_documents
from vector_store import VectorStore
from hybrid_search import HybridSearch
from reranker import LLMReranker

load_dotenv()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"


def gemini_chat(api_key, messages, system_prompt="", temperature=0.1, max_tokens=1000):
    """Call Gemini REST API directly for chat generation."""
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    resp = requests.post(GEMINI_URL, params={"key": api_key},
                         headers={"Content-Type": "application/json"},
                         json=payload, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class RAGEngine:
    def __init__(self, openai_api_key=None, model="gemini-3.6-flash",
                 rerank_model="gemini-3.6-flash", collection_name="rag_docs",
                 persist_dir="./chroma_db"):
        self.api_key = openai_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required! Set GEMINI_API_KEY in .env file.")
        self.model = model
        print("\n🚀 Initializing RAG Engine...\n")
        self.vector_store = VectorStore(api_key=self.api_key,
                                        collection_name=collection_name,
                                        persist_dir=persist_dir)
        self.hybrid_search = HybridSearch(self.vector_store)
        self.reranker = LLMReranker(self.api_key, model=rerank_model)
        if self.vector_store.count() > 0:
            self.hybrid_search.build_bm25_index()
        self.conversation_history = []
        print("\n✅ RAG Engine ready!\n")

    def add_documents(self, file_paths, chunk_size=500):
        print(f"\n📚 Adding {len(file_paths)} document(s)...\n")
        chunks = process_documents(file_paths, chunk_size=chunk_size)
        if not chunks:
            print("❌ No chunks created.")
            return
        self.vector_store.add_documents(chunks)
        self.hybrid_search.build_bm25_index()
        print(f"\n✅ Knowledge base updated! Total chunks: {self.vector_store.count()}")

    def retrieve(self, query, n_candidates=10, n_final=4):
        candidates = self.hybrid_search.search(query, n_results=n_candidates)
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, top_k=n_final)

    def generate(self, query, context_chunks):
        if not context_chunks:
            return ("I could not find relevant information. Please upload relevant documents.", [])
        context_parts = []
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
            if chunk["source"] not in citations:
                citations.append(chunk["source"])
        context_text = "\n\n---\n\n".join(context_parts)
        system_prompt = "You are a helpful assistant. Answer ONLY from the provided context. Cite sources as [Source N]."
        user_prompt = f"Context:\n\n{context_text}\n\n---\n\nQuestion: {query}"
        self.conversation_history.append({"role": "user", "content": user_prompt})
        answer = gemini_chat(self.api_key, self.conversation_history[-6:],
                             system_prompt=system_prompt, temperature=0.1, max_tokens=1000)
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer, citations

    def evaluate_answer(self, query, answer, context_chunks):
        context_text = "\n\n".join([c["text"] for c in context_chunks])
        eval_prompt = f"""Evaluate this RAG answer. Respond with ONLY valid JSON.

Question: {query}
Context: {context_text[:1500]}
Answer: {answer}

JSON format:
{{
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "completeness": <1-5>,
    "overall": <1-5>,
    "feedback": "<one sentence>"
}}"""
        try:
            result = gemini_chat(self.api_key, [{"role": "user", "content": eval_prompt}],
                                 temperature=0, max_tokens=200)
            start = result.find("{")
            end = result.rfind("}") + 1
            return json.loads(result[start:end])
        except Exception as e:
            return {"faithfulness": 0, "relevance": 0, "completeness": 0,
                    "overall": 0, "feedback": f"Evaluation failed: {e}"}

    def ask(self, query, n_candidates=10, n_final=4, evaluate=False):
        print(f"\n❓ Query: {query}\n")
        context_chunks = self.retrieve(query, n_candidates=n_candidates, n_final=n_final)
        answer, citations = self.generate(query, context_chunks)
        result = {"query": query, "answer": answer, "citations": citations,
                  "context_chunks": context_chunks, "num_sources_found": len(context_chunks)}
        if evaluate and context_chunks:
            result["evaluation"] = self.evaluate_answer(query, answer, context_chunks)
        return result

    def clear_history(self):
        self.conversation_history = []
        print("🔄 Conversation history cleared.")
