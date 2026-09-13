"""
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
