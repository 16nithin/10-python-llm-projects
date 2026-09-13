"""
cli.py
------
Command-line interface for the RAG system.
Use this to test the system without a browser.

Usage:
    python cli.py --add path/to/doc.pdf path/to/doc2.txt
    python cli.py --query "What is the main topic?"
    python cli.py  # Interactive mode
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()


def print_result(result: dict, show_sources: bool = True, show_eval: bool = True):
    """Pretty-print a RAG result to the terminal."""
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])

    if result["citations"]:
        print(f"\n📚 Sources: {', '.join(result['citations'])}")

    if show_sources and result.get("context_chunks"):
        print("\n" + "-"*60)
        print("SOURCE CHUNKS USED:")
        print("-"*60)
        for i, chunk in enumerate(result["context_chunks"], 1):
            print(f"\n[{i}] From: {chunk['source']}")
            score = chunk.get('llm_relevance_score', chunk.get('fusion_score', 0))
            print(f"    Relevance: {score:.1f}/10")
            print(f"    Text: {chunk['text'][:200]}...")

    if show_eval and result.get("evaluation"):
        eval_data = result["evaluation"]
        print("\n" + "-"*60)
        print("EVALUATION SCORES:")
        print("-"*60)
        print(f"  Faithfulness : {eval_data.get('faithfulness', '?')}/5")
        print(f"  Relevance    : {eval_data.get('relevance', '?')}/5")
        print(f"  Completeness : {eval_data.get('completeness', '?')}/5")
        print(f"  Overall      : {eval_data.get('overall', '?')}/5")
        print(f"  Feedback     : {eval_data.get('feedback', '')}")


def main():
    parser = argparse.ArgumentParser(description="Production RAG System CLI")
    parser.add_argument("--add", nargs="+", help="Add documents to knowledge base")
    parser.add_argument("--query", type=str, help="Ask a single question")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--no-sources", action="store_true", help="Hide source chunks")
    parser.add_argument("--clear", action="store_true", help="Clear the knowledge base")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set.")
        print("   1. Get a free key at: https://aistudio.google.com")
        print("   2. Add it to your .env file: GEMINI_API_KEY=AIza...")
        sys.exit(1)

    # Initialize the RAG engine
    print("\n🚀 Initializing RAG System...")
    engine = RAGEngine(openai_api_key=api_key)

    # Clear knowledge base
    if args.clear:
        engine.vector_store.clear()
        print("✅ Knowledge base cleared.")
        return

    # Add documents
    if args.add:
        engine.add_documents(args.add)
        return

    # Single query mode
    if args.query:
        result = engine.ask(args.query, evaluate=not args.no_eval)
        print_result(result, show_sources=not args.no_sources, show_eval=not args.no_eval)
        return

    # Interactive chat mode
    print("\n💬 Interactive RAG Chat Mode")
    print("   Type your question and press Enter")
    print("   Type 'exit' to quit, 'clear' to reset history\n")

    while True:
        try:
            query = input("\n🙋 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Bye!")
            break

        if not query:
            continue
        elif query.lower() == "exit":
            print("\n👋 Bye!")
            break
        elif query.lower() == "clear":
            engine.clear_history()
            continue

        result = engine.ask(query, evaluate=not args.no_eval)
        print_result(result, show_sources=not args.no_sources, show_eval=not args.no_eval)


if __name__ == "__main__":
    main()
