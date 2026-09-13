"""
Run this from inside your rag_system folder:
    python patch_all.py
This fixes BOTH chat generation and reranking to use the Gemini REST API directly.
"""

# ─── rag_engine.py ───────────────────────────────────────────────────────────
rag_engine = '''"""
rag_engine.py - RAG pipeline using Gemini REST API directly (no OpenAI client)
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

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def gemini_chat(api_key: str, messages: List[Dict], system_prompt: str = "",
                temperature: float = 0.1, max_tokens: int = 1000) -> str:
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

    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


class RAGEngine:
    def __init__(self, openai_api_key=None, model="gemini-1.5-flash",
                 rerank_model="gemini-1.5-flash", collection_name="rag_docs",
                 persist_dir="./chroma_db"):
        self.api_key = openai_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required! Set GEMINI_API_KEY in .env file.")
        self.model = model

        print("\\n🚀 Initializing RAG Engine...\\n")
        self.vector_store = VectorStore(api_key=self.api_key,
                                        collection_name=collection_name,
                                        persist_dir=persist_dir)
        self.hybrid_search = HybridSearch(self.vector_store)
        self.reranker = LLMReranker(self.api_key, model=rerank_model)

        if self.vector_store.count() > 0:
            self.hybrid_search.build_bm25_index()

        self.conversation_history: List[Dict[str, str]] = []
        print("\\n✅ RAG Engine ready!\\n")

    def add_documents(self, file_paths: List[str], chunk_size: int = 500) -> None:
        print(f"\\n📚 Adding {len(file_paths)} document(s) to knowledge base...\\n")
        chunks = process_documents(file_paths, chunk_size=chunk_size)
        if not chunks:
            print("❌ No chunks created.")
            return
        self.vector_store.add_documents(chunks)
        self.hybrid_search.build_bm25_index()
        print(f"\\n✅ Knowledge base updated! Total chunks: {self.vector_store.count()}")

    def retrieve(self, query: str, n_candidates: int = 10, n_final: int = 4) -> List[Dict]:
        candidates = self.hybrid_search.search(query, n_results=n_candidates)
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, top_k=n_final)

    def generate(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[str]]:
        if not context_chunks:
            return ("I couldn\'t find relevant information in the knowledge base. "
                    "Please upload relevant documents first.", [])

        context_parts = []
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk[\'source\']}]\\n{chunk[\'text\']}")
            if chunk["source"] not in citations:
                citations.append(chunk["source"])

        context_text = "\\n\\n---\\n\\n".join(context_parts)

        system_prompt = """You are a helpful knowledge assistant. Answer questions based ONLY on the provided context.
Rules:
1. Only use information from the provided context
2. Cite your sources using [Source N] notation inline
3. If the context doesn\'t contain enough information, say so clearly
4. Be concise but complete"""

        user_prompt = f"""Context from knowledge base:

{context_text}

---

Question: {query}

Please answer based on the context above, citing your sources."""

        self.conversation_history.append({"role": "user", "content": user_prompt})
        recent_history = self.conversation_history[-6:]

        answer = gemini_chat(self.api_key, recent_history, system_prompt=system_prompt,
                             temperature=0.1, max_tokens=1000)
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer, citations

    def evaluate_answer(self, query: str, answer: str, context_chunks: List[Dict]) -> Dict:
        context_text = "\\n\\n".join([c["text"] for c in context_chunks])
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
            # Extract JSON from response
            start = result.find("{")
            end = result.rfind("}") + 1
            return json.loads(result[start:end])
        except Exception as e:
            return {"faithfulness": 0, "relevance": 0, "completeness": 0,
                    "overall": 0, "feedback": f"Evaluation failed: {e}"}

    def ask(self, query: str, n_candidates: int = 10, n_final: int = 4,
            evaluate: bool = False) -> Dict:
        print(f"\\n❓ Query: {query}\\n")
        print("🔍 Searching knowledge base...")
        context_chunks = self.retrieve(query, n_candidates=n_candidates, n_final=n_final)
        print("💭 Generating answer...")
        answer, citations = self.generate(query, context_chunks)

        result = {
            "query": query, "answer": answer, "citations": citations,
            "context_chunks": context_chunks, "num_sources_found": len(context_chunks)
        }
        if evaluate and context_chunks:
            print("📊 Evaluating answer quality...")
            result["evaluation"] = self.evaluate_answer(query, answer, context_chunks)
        return result

    def clear_history(self) -> None:
        self.conversation_history = []
        print("🔄 Conversation history cleared.")
'''

# ─── reranker.py ─────────────────────────────────────────────────────────────
reranker = '''"""
reranker.py - LLM reranker using Gemini REST API directly
"""

import json
import requests
from typing import List, Dict, Any

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


class LLMReranker:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model

    def _call_gemini(self, prompt: str) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 100}
        }
        resp = requests.post(
            GEMINI_URL,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        if resp.status_code != 200:
            raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        scored = []
        for chunk in candidates:
            prompt = f"""Rate how relevant this text is to the question. Reply with ONLY a number 0-10.

Question: {query}

Text: {chunk["text"][:500]}

Score (0-10):"""
            try:
                score_text = self._call_gemini(prompt)
                score = float("".join(c for c in score_text if c.isdigit() or c == "."))
                score = max(0.0, min(10.0, score))
            except Exception:
                score = 5.0

            if score >= 3:
                chunk["llm_relevance_score"] = score
                scored.append(chunk)

        scored.sort(key=lambda x: x.get("llm_relevance_score", 0), reverse=True)
        return scored[:top_k]
'''

# Write both files
with open("rag_engine.py", "w", encoding="utf-8") as f:
    f.write(rag_engine)
print("✅ rag_engine.py patched")

with open("reranker.py", "w", encoding="utf-8") as f:
    f.write(reranker)
print("✅ reranker.py patched")

print("")
print("Both files updated! Now run:")
print("   streamlit run app.py")
