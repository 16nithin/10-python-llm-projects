"""
Run this from inside your rag_system folder:
    python fix_reranker_nollm.py

This replaces the LLM reranker with a simple score-based reranker.
Instead of calling Gemini 10 times per question (expensive!), it just
sorts the results by their existing search scores. You get 1 API call
per question instead of 11 — meaning 20 questions per day on free tier.
"""

reranker = '''"""
reranker.py - Simple score-based reranker (no API calls needed)
Sorts candidates by their existing hybrid search scores.
This saves your daily API quota for actual question answering!
"""

from typing import List, Dict, Any


class LLMReranker:
    """
    Reranker that uses existing search scores instead of calling the API.
    This preserves your 20 daily API calls for generating answers.
    """
    def __init__(self, api_key: str = None, model: str = "gemini-3.6-flash"):
        self.api_key = api_key
        self.model = model
        print("✅ Score-based reranker ready (no API calls needed)")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Rerank candidates using their existing search scores.
        No API call needed — just sorts by the score hybrid search already computed.
        """
        if not candidates:
            return []

        # Sort by existing score (hybrid search already computed a good score)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return sorted_candidates[:top_k]
'''

with open("reranker.py", "w", encoding="utf-8") as f:
    f.write(reranker)

print("✅ reranker.py updated — now uses 0 API calls for reranking!")
print("   Each question now costs exactly 1 API call.")
print("   With 20 free calls/day, you get 20 questions per day.")
print("")
print("Now run:  streamlit run app.py")
