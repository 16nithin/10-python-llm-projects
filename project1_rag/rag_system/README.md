# 🧠 Project 1: Production RAG System

A knowledge assistant that answers questions about your documents using **hybrid search**, **reranking**, **citations**, and **self-evaluation**.

## 🏗️ Architecture

```
Your Documents (PDF/DOCX/TXT)
        ↓
[Document Processor]  → Splits into 500-char chunks
        ↓
[Vector Store]        → Embeds chunks (sentence-transformers) → ChromaDB
        ↓
User Question
        ↓
[Hybrid Search]       → Semantic Search + BM25 Keyword Search → 10 candidates
        ↓
[LLM Reranker]        → GPT-4o-mini scores each candidate → Top 4 chunks
        ↓
[Answer Generation]   → GPT-4o-mini generates answer with citations
        ↓
[Evaluator]           → GPT-4o-mini grades: Faithfulness, Relevance, Completeness
```

## 📁 File Structure

```
rag_system/
├── requirements.txt          # Python dependencies
├── .env.example              # Template for your API key
├── document_processor.py     # Load & chunk PDF/DOCX/TXT files
├── vector_store.py           # ChromaDB wrapper for semantic search
├── hybrid_search.py          # BM25 + semantic search + RRF fusion
├── reranker.py               # LLM-based result reranking
├── rag_engine.py             # Main pipeline (ties everything together)
├── app.py                    # Streamlit web UI
├── cli.py                    # Command-line interface
└── README.md                 # This file
```

## ⚡ Quick Start

### 1. Set up environment

```bash
cd rag_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...
```

### 3. Run the web UI

```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### 4. OR use the command line

```bash
# Add documents
python cli.py --add my_document.pdf notes.txt

# Ask a question
python cli.py --query "What are the main points in the document?"

# Interactive chat mode
python cli.py
```

## 🧠 Key Concepts Learned

| Concept | What it does | File |
|---------|-------------|------|
| **Chunking** | Splits large docs into searchable pieces | `document_processor.py` |
| **Embeddings** | Converts text to numbers capturing meaning | `vector_store.py` |
| **Semantic Search** | Finds text by meaning, not keywords | `vector_store.py` |
| **BM25 Keyword Search** | Exact keyword matching (like Google) | `hybrid_search.py` |
| **RRF Fusion** | Merges two ranked lists intelligently | `hybrid_search.py` |
| **LLM Reranking** | GPT scores relevance of candidates | `reranker.py` |
| **RAG Generation** | GPT answers using retrieved context | `rag_engine.py` |
| **Self-Evaluation** | GPT grades its own answer quality | `rag_engine.py` |

## 🔧 Customization Tips

- **Chunk size**: Increase to 800-1000 for longer context, decrease to 200-300 for more precise retrieval
- **Model**: Change `gpt-4o-mini` to `gpt-4o` for better quality (more expensive)
- **n_candidates**: Increase from 10 to 20 for better recall (slightly slower)
- **Embedding model**: Try `all-mpnet-base-v2` for better quality embeddings
