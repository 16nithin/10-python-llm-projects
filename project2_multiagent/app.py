"""
app.py - Streamlit UI (Groq/Llama version - FREE!)
Run with: streamlit run app.py
"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Research System")
st.caption("Powered by Groq + Llama 3.1 · FREE · 14,400 requests/day · 3 specialized agents")

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
**3 specialized AI agents** work as a team (powered by Groq — completely free!):

| Agent | Role |
|-------|------|
| 🔍 Researcher | Searches Wikipedia, structures raw facts |
| ✅ Verifier | Checks accuracy, flags uncertain claims |
| 📝 Summarizer | Writes polished final report |

**Free tier: 14,400 requests/day** — way more than enough!
Get your free key at **console.groq.com**
    """)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free key at console.groq.com — no credit card needed!"
    )
    st.success("✅ Completely FREE\n14,400 requests/day\nNo credit card needed")
    st.markdown("### 🎯 Example Topics")
    for ex in ["Artificial Intelligence", "Climate Change", "Quantum Computing",
                "The Roman Empire", "Black Holes", "Machine Learning", "Python Programming"]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["topic_input"] = ex

col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("Research Topic",
                           placeholder="e.g., Machine Learning, Climate Change...",
                           key="topic_input", label_visibility="collapsed")
with col2:
    go = st.button("🚀 Research", type="primary", use_container_width=True)

if go:
    if not api_key:
        st.error("❌ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
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
            for col, label in zip(cols, ["🔍 Researcher", "✅ Verifier", "📝 Summarizer"]):
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
            else:
                st.markdown("---")
                st.markdown("## 📋 Final Research Report")
                st.markdown(results["final_report"])
                st.markdown("---")
                st.markdown("## 🔍 Agent Details (Behind the Scenes)")
                icons = {"Research": "🔍", "Verification": "✅", "Summary": "📝"}
                for a in results["agents"]:
                    with st.expander(f"{icons.get(a['step'],'🤖')} {a['name']} — {a['step']}"):
                        st.markdown(a["output"])
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.markdown("---")
    st.info("👆 Enter any topic above and click Research. Completely free — no limits!")
