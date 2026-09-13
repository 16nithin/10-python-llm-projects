"""
app.py
------
Streamlit web UI for the Production RAG System.
Run with: streamlit run app.py

Uses Google Gemini API (FREE) — get your key at aistudio.google.com
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Production RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag_engine import RAGEngine


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG System")
    st.markdown("*Powered by Google Gemini (Free)*")
    st.markdown("---")

    # API Key input
    st.subheader("🔑 Setup")
    st.markdown("Get a **free** API key at [aistudio.google.com](https://aistudio.google.com)")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Your free Google Gemini API key"
    )

    if api_key:
        st.session_state["gemini_api_key"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown("---")

    # Document upload
    st.subheader("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, Word, or text files"
    )

    if uploaded_files and api_key:
        if st.button("📥 Index Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                final_paths = []
                try:
                    for uploaded_file in uploaded_files:
                        suffix = Path(uploaded_file.name).suffix
                        orig_name = uploaded_file.name
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix, mode='wb'
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        # Rename to original filename so citations look nice
                        new_path = os.path.join(os.path.dirname(tmp_path), orig_name)
                        os.rename(tmp_path, new_path)
                        final_paths.append(new_path)

                    engine = RAGEngine(openai_api_key=api_key)
                    engine.add_documents(final_paths)
                    st.session_state["rag_engine"] = engine
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    for path in final_paths:
                        try:
                            os.unlink(path)
                        except:
                            pass

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    show_evaluation = st.toggle("Show answer evaluation", value=True)
    show_sources = st.toggle("Show source chunks", value=True)
    n_candidates = st.slider("Search candidates", 5, 20, 10)
    n_final = st.slider("Context chunks", 2, 8, 4)

    if st.session_state.get("chat_history"):
        st.markdown("---")
        st.subheader("💬 History")
        for i, (q, _) in enumerate(st.session_state.get("chat_history", [])):
            st.caption(f"{i+1}. {q[:50]}...")


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────
st.title("🧠 Production RAG System")
st.markdown("*Ask questions about your documents — answers come with citations and quality scores.*")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Render existing chat
for question, result in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        if result.get("citations"):
            st.caption(f"📚 Sources: {', '.join(result['citations'])}")

        if show_sources and result.get("context_chunks"):
            with st.expander("📄 Source chunks used"):
                for i, chunk in enumerate(result["context_chunks"], 1):
                    st.markdown(f"**[{i}] {chunk['source']}**")
                    preview = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                    st.text(preview)
                    score = chunk.get('llm_relevance_score', chunk.get('fusion_score', 0))
                    st.caption(f"Relevance: {score:.1f}/10")
                    if i < len(result["context_chunks"]):
                        st.divider()

        if show_evaluation and result.get("evaluation"):
            ev = result["evaluation"]
            with st.expander("📊 Answer Quality Scores"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness", f"{ev.get('faithfulness', 0)}/5")
                c2.metric("Relevance", f"{ev.get('relevance', 0)}/5")
                c3.metric("Completeness", f"{ev.get('completeness', 0)}/5")
                c4.metric("Overall", f"{ev.get('overall', 0)}/5")
                st.caption(ev.get('feedback', ''))

# Show appropriate prompt based on state
if not api_key:
    st.warning("👈 Enter your free Gemini API key in the sidebar to get started.")
elif "rag_engine" not in st.session_state:
    st.info("👈 Upload documents and click 'Index Documents' to build your knowledge base.")
else:
    query = st.chat_input("Ask a question about your documents...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    engine = st.session_state["rag_engine"]
                    result = engine.ask(
                        query=query,
                        n_candidates=n_candidates,
                        n_final=n_final,
                        evaluate=show_evaluation
                    )

                    st.write(result["answer"])

                    if result.get("citations"):
                        st.caption(f"📚 Sources: {', '.join(result['citations'])}")

                    if show_sources and result.get("context_chunks"):
                        with st.expander("📄 Source chunks used"):
                            for i, chunk in enumerate(result["context_chunks"], 1):
                                st.markdown(f"**[{i}] {chunk['source']}**")
                                preview = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                                st.text(preview)
                                score = chunk.get('llm_relevance_score', chunk.get('fusion_score', 0))
                                st.caption(f"Relevance: {score:.1f}/10")
                                if i < len(result["context_chunks"]):
                                    st.divider()

                    if show_evaluation and result.get("evaluation"):
                        ev = result["evaluation"]
                        with st.expander("📊 Answer Quality Scores"):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Faithfulness", f"{ev.get('faithfulness', 0)}/5")
                            c2.metric("Relevance", f"{ev.get('relevance', 0)}/5")
                            c3.metric("Completeness", f"{ev.get('completeness', 0)}/5")
                            c4.metric("Overall", f"{ev.get('overall', 0)}/5")
                            st.caption(ev.get('feedback', ''))

                    st.session_state["chat_history"].append((query, result))

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Check your Gemini API key and make sure documents are indexed.")
