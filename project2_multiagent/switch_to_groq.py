"""
switch_to_groq.py
Run from inside your project2_multiagent folder:
    python switch_to_groq.py

Switches to Groq API — FREE, 14,400 requests/day, no credit card needed.
Uses Llama 3.1 (fast, capable, completely free).
Get your free key at: console.groq.com
"""

import os

agents = '''"""
agents.py - Three specialized AI agents using Groq API (FREE!)
Groq runs Llama 3.1 — fast, free, 14,400 requests/day
"""
import requests, os
from dotenv import load_dotenv
from tools import gather_research

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # Fast, free Llama model


def call_groq(api_key, system_prompt, user_message, temperature=0.3, max_tokens=1500):
    """
    Call Groq API — it uses the same format as OpenAI, very simple.
    Authorization: Bearer YOUR_KEY (in header)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


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
        print(f"  {self.name}: Analyzing with Groq/Llama...")
        output = call_groq(
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
        output = call_groq(
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
        output = call_groq(
            self.api_key, self.system_prompt,
            f\'Create a research summary about "{topic}":\\n\\nRESEARCH NOTES:\\n{verification_output["research_notes"]}\\n\\nVERIFICATION:\\n{verification_output["output"]}\',
            temperature=0.4, max_tokens=1500)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "output": output}
'''

orchestrator = '''"""
orchestrator.py - Coordinates all three agents
"""
import os
from dotenv import load_dotenv
from agents import ResearcherAgent, VerifierAgent, SummarizerAgent

load_dotenv()

class MultiAgentOrchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required! Get a free one at console.groq.com")
        self.researcher = ResearcherAgent(self.api_key)
        self.verifier = VerifierAgent(self.api_key)
        self.summarizer = SummarizerAgent(self.api_key)
        print("✅ Multi-Agent Research System ready! (powered by Groq/Llama — FREE)")

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

app = '''"""
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
    st.success("✅ Completely FREE\\n14,400 requests/day\\nNo credit card needed")
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
    st.info("👆 Enter any topic above and click Research. Completely free — no limits!")
'''

for fname, content in [("agents.py", agents), ("orchestrator.py", orchestrator), ("app.py", app)]:
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Updated: {fname}")

# Update .env
env_content = ""
if os.path.exists(".env"):
    with open(".env", "r") as f:
        env_content = f.read()

if "GROQ_API_KEY" not in env_content:
    with open(".env", "a") as f:
        f.write("\nGROQ_API_KEY=your_groq_key_here\n")
    print("✅ Updated .env — add your Groq key")

print("""
🎉 Switched to Groq (FREE)!

1. Get your free key at: console.groq.com
2. Open .env and set:  GROQ_API_KEY=gsk_your_key_here
3. Run: streamlit run app.py

Free tier: 14,400 requests/day — unlimited for learning!
""")
