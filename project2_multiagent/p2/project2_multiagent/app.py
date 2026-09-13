"""
app.py - Streamlit UI for Multi-Agent Research System
Run with: streamlit run app.py
"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Research System")
st.caption("3 specialized AI agents research, verify, and summarize any topic collaboratively")

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
**3 specialized AI agents** work as a team — each with one job:

| Agent | Role | API Calls |
|-------|------|-----------|
| 🔍 Researcher | Searches Wikipedia, structures raw facts | 1 |
| ✅ Verifier | Checks accuracy, flags uncertain claims | 1 |
| 📝 Summarizer | Writes polished final report | 1 |

**Total: 3 API calls per session** → 6 research sessions/day on free tier.
    """)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password",
                             value=os.getenv("GEMINI_API_KEY", ""),
                             help="Get free key at aistudio.google.com/apikey")
    st.info("3 API calls per research session\nFree tier: 20/day = 6 sessions/day")
    st.markdown("### 🎯 Example Topics")
    for ex in ["Artificial Intelligence", "Climate Change", "Quantum Computing",
                "The Roman Empire", "Black Holes", "Machine Learning"]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["topic_input"] = ex

col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("Research Topic", placeholder="e.g., Artificial Intelligence...",
                           key="topic_input", label_visibility="collapsed")
with col2:
    go = st.button("🚀 Research", type="primary", use_container_width=True)

if go:
    if not api_key:
        st.error("❌ Please enter your Gemini API key in the sidebar.")
    elif not topic.strip():
        st.error("❌ Please enter a research topic.")
    else:
        try:
            from orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator(api_key=api_key)
            st.markdown("---")
            st.subheader(f"🔬 Researching: **{topic}**")
            cols = st.columns(3)
            boxes = []
            labels = ["🔍 Researcher", "✅ Verifier", "📝 Summarizer"]
            for i, (col, label) in enumerate(zip(cols, labels)):
                with col:
                    st.markdown(f"#### {label}")
                    b = st.empty()
                    b.info("⏳ Waiting...")
                    boxes.append(b)
            prog = st.progress(0, text="Starting...")

            def update(step, name, out):
                boxes[step-1].success("✓ Done")
                prog.progress(step/3, text=f"Step {step}/3 complete")

            with st.spinner("Agents working..."):
                results = orch.research(topic, progress_callback=update)

            prog.progress(1.0, text="✅ All agents complete!")

            if "error" in results:
                st.error(f"❌ Error: {results['error']}")
                if "429" in results["error"]:
                    st.warning("⚠️ Daily quota reached (20 calls = 6 sessions). Wait until tomorrow or use a new API key.")
            else:
                st.markdown("---")
                st.markdown("## 📋 Final Research Report")
                st.markdown(results["final_report"])
                st.markdown("---")
                st.markdown("## 🔍 Agent Details (Behind the Scenes)")
                icons = {"Research": "🔍", "Verification": "✅", "Summary": "📝"}
                for a in results["agents"]:
                    with st.expander(f"{icons.get(a['step'], '🤖')} {a['name']} — {a['step']}"):
                        st.markdown(a["output"])
        except Exception as e:
            st.error(f"❌ Error: {e}")
            if "429" in str(e):
                st.warning("⚠️ Daily quota reached. Wait until tomorrow or use a new API key.")
else:
    st.markdown("---")
    st.info("👆 Enter any topic above and click Research. The 3 agents will work together automatically.")
