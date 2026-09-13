"""
switch_to_claude_p2.py
Run this from inside your project2_multiagent folder:
    python switch_to_claude_p2.py

Switches all files from Gemini API → Claude API.
Uses claude-3-5-haiku (cheapest Claude model, ~$0.001 per request)
"""

import os

# ── agents.py ─────────────────────────────────────────────────────────────────
agents = '''"""
agents.py - Three specialized AI agents using Claude API
"""
import requests, os
from dotenv import load_dotenv
from tools import gather_research

load_dotenv()

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-3-5-haiku-20241022"  # Cheapest Claude model


def call_claude(api_key, system_prompt, user_message, temperature=0.3, max_tokens=1500):
    """
    Call Claude API directly using requests.
    Claude uses a different format than Gemini:
    - API key goes in the header (x-api-key), not as a URL param
    - system prompt is a separate field, not inside contents
    - response is in content[0]["text"] not candidates[0]...
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }
    resp = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Claude API error {resp.status_code}: {resp.text}")
    return resp.json()["content"][0]["text"]


class ResearcherAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.name = "🔍 Researcher Agent"
        self.system_prompt = """You are a Research Specialist AI. Extract key facts from provided data.

Format output as:
## KEY FACTS
(bullet points)

## BACKGROUND CONTEXT
(1-2 paragraphs)

## IMPORTANT DETAILS
(dates, numbers, names)

## GAPS IN RESEARCH
(what is missing or unclear)

Only use information from the provided data."""

    def run(self, topic):
        print(f"  {self.name}: Gathering data about \'{topic}\'...")
        raw_data = gather_research(topic)
        print(f"  {self.name}: Analyzing with Claude...")
        output = call_claude(
            self.api_key, self.system_prompt,
            f\'Please analyze this research data about "{topic}":\\n\\n{raw_data}\',
            temperature=0.2, max_tokens=1200)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "raw_data": raw_data, "output": output}


class VerifierAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.name = "✅ Verifier Agent"
        self.system_prompt = """You are a Fact-Checking Specialist AI. Verify research notes critically.

Format output as:
## VERIFIED FACTS ✓
(reliable, well-supported facts)

## UNCERTAIN CLAIMS ⚠️
(claims needing caveats)

## CONTRADICTIONS OR ISSUES ❌
(contradictions found, or "None found")

## RELIABILITY ASSESSMENT
(rate quality 1-10 with brief explanation)

## RECOMMENDED CAVEATS
(what readers should know about limitations)"""

    def run(self, research_output):
        print(f"  {self.name}: Verifying findings...")
        topic = research_output["topic"]
        output = call_claude(
            self.api_key, self.system_prompt,
            f\'Verify these research notes about "{topic}":\\n\\n{research_output["output"]}\\n\\nORIGINAL DATA:\\n{research_output["raw_data"][:2000]}\',
            temperature=0.1, max_tokens=1000)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic,
                "research_notes": research_output["output"], "output": output}


class SummarizerAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.name = "📝 Summarizer Agent"
        self.system_prompt = """You are a Professional Research Writer AI. Create polished summaries.

Format output as:
# [Topic]: Research Summary

## Overview
(2-3 sentence introduction)

## Key Findings
(3-5 bullet points)

## Detailed Analysis
(2-3 paragraphs)

## Reliability Notes
(confidence level and caveats)

## Bottom Line
(1-2 sentences: most important takeaway)

Write clearly for a general audience."""

    def run(self, verification_output):
        print(f"  {self.name}: Writing final report...")
        topic = verification_output["topic"]
        output = call_claude(
            self.api_key, self.system_prompt,
            f\'Create a research summary about "{topic}":\\n\\nRESEARCH NOTES:\\n{verification_output["research_notes"]}\\n\\nVERIFICATION:\\n{verification_output["output"]}\',
            temperature=0.4, max_tokens=1500)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "output": output}
'''

# ── orchestrator.py ───────────────────────────────────────────────────────────
orchestrator = '''"""
orchestrator.py - Coordinates all three agents
"""
import os
from dotenv import load_dotenv
from agents import ResearcherAgent, VerifierAgent, SummarizerAgent

load_dotenv()

class MultiAgentOrchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key required! Set ANTHROPIC_API_KEY in .env file.")
        self.researcher = ResearcherAgent(self.api_key)
        self.verifier = VerifierAgent(self.api_key)
        self.summarizer = SummarizerAgent(self.api_key)
        print("✅ Multi-Agent Research System ready! (powered by Claude)")

    def research(self, topic, progress_callback=None):
        print(f"\\n🚀 Researching: \'{topic}\'\\n")
        results = {"topic": topic, "agents": [], "final_report": ""}
        try:
            print("STEP 1/3: Researcher")
            r = self.researcher.run(topic)
            results["agents"].append({"name": self.researcher.name, "output": r["output"], "step": "Research"})
            if progress_callback: progress_callback(1, self.researcher.name, r["output"])

            print("\\nSTEP 2/3: Verifier")
            v = self.verifier.run(r)
            results["agents"].append({"name": self.verifier.name, "output": v["output"], "step": "Verification"})
            if progress_callback: progress_callback(2, self.verifier.name, v["output"])

            print("\\nSTEP 3/3: Summarizer")
            s = self.summarizer.run(v)
            results["agents"].append({"name": self.summarizer.name, "output": s["output"], "step": "Summary"})
            results["final_report"] = s["output"]
            if progress_callback: progress_callback(3, self.summarizer.name, s["output"])

            print("\\n✅ Research complete!")
            return results
        except Exception as e:
            results["error"] = str(e)
            return results
'''

# ── app.py ────────────────────────────────────────────────────────────────────
app = '''"""
app.py - Streamlit UI (Claude version)
Run with: streamlit run app.py
"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Research System")
st.caption("Powered by Claude claude-3-5-haiku · 3 specialized agents · research, verify, summarize")

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
**3 specialized Claude agents** work as a team:

| Agent | Role | Cost |
|-------|------|------|
| 🔍 Researcher | Searches Wikipedia, structures raw facts | ~$0.001 |
| ✅ Verifier | Checks accuracy, flags uncertain claims | ~$0.001 |
| 📝 Summarizer | Writes polished final report | ~$0.001 |

**Total: ~$0.003 per research session** · $5 free credit = ~1,600 sessions
    """)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Get your key at console.anthropic.com → API Keys"
    )
    st.success("$5 free credit on signup\\n~$0.003 per research session\\n~1,600 sessions for $5")
    st.markdown("### 🎯 Example Topics")
    for ex in ["Artificial Intelligence", "Climate Change", "Quantum Computing",
                "The Roman Empire", "Black Holes", "Machine Learning"]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["topic_input"] = ex

col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("Research Topic",
                           placeholder="e.g., Artificial Intelligence...",
                           key="topic_input", label_visibility="collapsed")
with col2:
    go = st.button("🚀 Research", type="primary", use_container_width=True)

if go:
    if not api_key:
        st.error("❌ Please enter your Anthropic API key in the sidebar.")
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
                st.error(f"❌ Error: {results[\'error\']}")
            else:
                st.markdown("---")
                st.markdown("## 📋 Final Research Report")
                st.markdown(results["final_report"])
                st.markdown("---")
                st.markdown("## 🔍 Agent Details (Behind the Scenes)")
                icons = {"Research": "🔍", "Verification": "✅", "Summary": "📝"}
                for a in results["agents"]:
                    with st.expander(f"{icons.get(a[\'step\'],\'🤖\')} {a[\'name\']} — {a[\'step\']}"):
                        st.markdown(a["output"])
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.markdown("---")
    st.info("👆 Enter any topic above and click Research.")
'''

# Write all files
for fname, content in [("agents.py", agents), ("orchestrator.py", orchestrator), ("app.py", app)]:
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Updated: {fname}")

# Create .env template
if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write("ANTHROPIC_API_KEY=your_key_here\n")
    print("✅ Created: .env  ← paste your Claude API key here")
else:
    # Update existing .env to replace Gemini key with Claude key
    with open(".env", "r") as f:
        env_content = f.read()
    if "ANTHROPIC_API_KEY" not in env_content:
        with open(".env", "a") as f:
            f.write("\nANTHROPIC_API_KEY=your_key_here\n")
        print("✅ Updated: .env  ← added ANTHROPIC_API_KEY line")

print("""
🎉 All files updated to use Claude API!

Next steps:
1. Open .env and paste your Claude API key:
   ANTHROPIC_API_KEY=sk-ant-...

2. Run the app:
   streamlit run app.py

3. Paste your key in the sidebar and research any topic.

Cost: ~$0.003 per research session · $5 free = ~1,600 sessions
""")
