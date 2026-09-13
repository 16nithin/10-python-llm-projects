"""
app.py - Streamlit Web UI for the Multi-Agent Research System
=============================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Research System")
st.caption("3 specialized AI agents research, verify, and summarize any topic collaboratively")

# ── How it works diagram ──────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    This system uses **3 specialized AI agents** working as a team:

    | Agent | Role | API Calls |
    |-------|------|-----------|
    | 🔍 Researcher | Searches Wikipedia, gathers and structures raw facts | 1 |
    | ✅ Verifier | Checks accuracy, flags uncertain claims | 1 |
    | 📝 Summarizer | Writes polished final report | 1 |

    **Total: 3 API calls per research session** (vs. 1 monolithic call).

    **Why multiple agents?**
    - Each agent is a specialist with a focused system prompt
    - The Researcher is optimized for finding facts
    - The Verifier is optimized for critical thinking
    - The Summarizer is optimized for clear writing
    - Together they produce better results than one LLM doing everything
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a free key at aistudio.google.com/apikey"
    )

    st.markdown("---")
    st.markdown("### 📊 API Usage")
    st.info("Each research session uses **3 API calls** (1 per agent).\n\nFree tier: 20 calls/day = **6 sessions/day**")

    st.markdown("---")
    st.markdown("### 🎯 Example Topics")
    examples = [
        "Artificial Intelligence",
        "Climate Change",
        "Quantum Computing",
        "The Roman Empire",
        "Black Holes",
        "Machine Learning",
        "Bitcoin and Cryptocurrency",
        "DNA and Genetics"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["topic_input"] = ex

# ── Main Input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g., Artificial Intelligence, Climate Change, The Roman Empire...",
        key="topic_input",
        label_visibility="collapsed"
    )
with col2:
    research_btn = st.button("🚀 Research", type="primary", use_container_width=True)

# ── Research Execution ────────────────────────────────────────────────────────
if research_btn:
    if not api_key:
        st.error("❌ Please enter your Gemini API key in the sidebar.")
    elif not topic.strip():
        st.error("❌ Please enter a research topic.")
    else:
        try:
            from orchestrator import MultiAgentOrchestrator

            orchestrator = MultiAgentOrchestrator(api_key=api_key)

            st.markdown("---")
            st.subheader(f"🔬 Researching: **{topic}**")

            # Create placeholder areas for live agent updates
            agent_cols = st.columns(3)
            status_boxes = []

            with agent_cols[0]:
                st.markdown("#### 🔍 Researcher")
                box1 = st.empty()
                box1.info("⏳ Waiting...")
                status_boxes.append(box1)

            with agent_cols[1]:
                st.markdown("#### ✅ Verifier")
                box2 = st.empty()
                box2.info("⏳ Waiting...")
                status_boxes.append(box2)

            with agent_cols[2]:
                st.markdown("#### 📝 Summarizer")
                box3 = st.empty()
                box3.info("⏳ Waiting...")
                status_boxes.append(box3)

            progress_bar = st.progress(0, text="Starting research...")

            # Callback to update UI as each agent completes
            def update_ui(step: int, agent_name: str, output: str):
                status_boxes[step - 1].success(f"✓ Done")
                progress_bar.progress(step / 3, text=f"Step {step}/3: {agent_name} complete")

            # Run all agents
            with st.spinner("Agents are working..."):
                results = orchestrator.research(topic, progress_callback=update_ui)

            progress_bar.progress(1.0, text="✅ All agents complete!")

            # ── Display Results ───────────────────────────────────────────
            if "error" in results:
                st.error(f"❌ Error: {results['error']}")
                if "429" in results["error"]:
                    st.warning("⚠️ API quota reached. You have 20 free calls/day (6 research sessions). Please wait until tomorrow or use a new API key.")
            else:
                st.markdown("---")

                # Final report prominently displayed
                st.markdown("## 📋 Final Research Report")
                st.markdown(results["final_report"])

                # Individual agent outputs in expanders
                st.markdown("---")
                st.markdown("## 🔍 Agent Workings (Behind the Scenes)")

                for agent_result in results["agents"]:
                    icon = {"Research": "🔍", "Verification": "✅", "Summary": "📝"}.get(agent_result["step"], "🤖")
                    with st.expander(f"{icon} {agent_result['name']} — {agent_result['step']}"):
                        st.markdown(agent_result["output"])

        except Exception as e:
            st.error(f"❌ Error: {e}")
            if "429" in str(e):
                st.warning("⚠️ API quota reached (20 calls/day = 6 research sessions). Please wait until tomorrow or get a new API key from a different Google account.")

# ── Empty state ───────────────────────────────────────────────────────────────
elif not research_btn:
    st.markdown("---")
    st.markdown("""
    ### 👆 Enter any topic above to start

    The 3 agents will work together:
    1. **Researcher** searches Wikipedia and gathers facts
    2. **Verifier** checks accuracy and flags uncertain claims
    3. **Summarizer** writes a polished report you can actually read

    No extra setup needed — just paste your API key and go!
    """)
