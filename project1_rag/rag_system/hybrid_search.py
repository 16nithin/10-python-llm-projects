"""
hybrid_search.py
----------------
Combines TWO types of search for better results:

1. SEMANTIC SEARCH (vector/embedding search):
   - Finds documents that MEAN the same thing, even if different words are used
   - Example: "car" matches "automobile"
   - Great for understanding intent

2. KEYWORD SEARCH (BM25 - Best Match 25):
   - Traditional search that finds exact word matches
   - Great for specific terms, names, codes, etc.
   - Example: "invoice #12345" matches exactly

Hybrid = best of both worlds! 🎯
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re


def tokenize(text: str) -> List[str]:
    """Simple tokenizer — split text into lowercase words."""
    return re.findall(r'\b\w+\b', text.lower())


class HybridSearch:
    """
    Combines semantic search (from VectorStore) with BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge the two result lists.
    """

    def __init__(self, vector_store):
        """
        Args:
            vector_store: An initialized VectorStore instance
        """
        self.vector_store = vector_store
        self.bm25 = None
        self.bm25_docs = []  # Keep track of documents for BM25

    def build_bm25_index(self) -> None:
        """
        Build the BM25 index from all documents in the vector store.
        Call this after adding documents, or when the store has changed.
        """
        print("🔍 Building BM25 keyword search index...")

        # Get all documents from the vector store
        self.bm25_docs = self.vector_store.get_all_documents()

        if not self.bm25_docs:
            print("⚠️  No documents to index for BM25.")
            return

        # Tokenize each document
        tokenized_docs = [tokenize(doc["text"]) for doc in self.bm25_docs]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✅ BM25 index built with {len(self.bm25_docs)} documents.")

    def keyword_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching.
        Returns ranked results with BM25 scores.
        """
        if not self.bm25:
            print("⚠️  BM25 index not built. Call build_bm25_index() first.")
            return []

        # Tokenize the query
        query_tokens = tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with documents and sort by score
        scored_docs = list(zip(scores, range(len(self.bm25_docs))))
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Return top n_results
        results = []
        for rank, (score, idx) in enumerate(scored_docs[:n_results]):
            if score > 0:  # Only include docs with actual matches
                results.append({
                    "text": self.bm25_docs[idx]["text"],
                    "source": self.bm25_docs[idx]["source"],
                    "score": float(score),
                    "rank": rank + 1,
                    "search_type": "keyword"
                })

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60,  # RRF constant — 60 is standard
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF formula: score = weight / (k + rank)
        Documents appearing in BOTH lists get a big boost!

        Args:
            semantic_results: Results from vector/semantic search
            keyword_results: Results from BM25 keyword search
            k: Smoothing constant (higher = less weight difference between ranks)
            semantic_weight: How much to trust semantic search (default 60%)
            keyword_weight: How much to trust keyword search (default 40%)
        """
        # Build a dict to accumulate scores: text -> {score, metadata}
        fusion_scores: Dict[str, Dict] = {}

        # Score semantic results
        for result in semantic_results:
            text = result["text"]
            rrf_score = semantic_weight / (k + result["rank"])

            if text not in fusion_scores:
                fusion_scores[text] = {
                    "text": text,
                    "source": result["source"],
                    "fusion_score": 0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "found_by": []
                }

            fusion_scores[text]["fusion_score"] += rrf_score
            fusion_scores[text]["semantic_rank"] = result["rank"]
            fusion_scores[text]["found_by"].append("semantic")

        # Score keyword results
        for result in keyword_results:
            text = result["text"]
            rrf_score = keyword_weight / (k + result["rank"])

            if text not in fusion_scores:
                fusion_scores[text] = {
                    "text": text,
                    "source": result["source"],
                    "fusion_score": 0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "found_by": []
                }

            fusion_scores[text]["fusion_score"] += rrf_score
            fusion_scores[text]["keyword_rank"] = result["rank"]
            fusion_scores[text]["found_by"].append("keyword")

        # Sort by fusion score (highest first)
        merged = sorted(
            fusion_scores.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )

        # Add final rank
        for i, result in enumerate(merged):
            result["rank"] = i + 1

        return merged

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Main search function — runs both search types and merges results.

        Args:
            query: The user's question
            n_results: How many final results to return

        Returns:
            Merged and ranked list of relevant document chunks
        """
        # Get more results from each search type before merging
        fetch_count = n_results * 2

        # Run both searches in parallel (conceptually)
        semantic_results = self.vector_store.semantic_search(query, n_results=fetch_count)
        keyword_results = self.keyword_search(query, n_results=fetch_count)

        # Merge using RRF
        merged = self.reciprocal_rank_fusion(semantic_results, keyword_results)

        # Return top results
        return merged[:n_results]
