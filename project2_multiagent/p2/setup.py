"""
setup.py - Creates the entire Project 2 folder structure
Run this from wherever you want the project to live:
    python setup.py
"""

import os

# All project files
files = {}

files["tools.py"] = '''"""
tools.py - Search tools for agents (FREE - no API key needed!)
"""
import requests

def search_wikipedia(query: str) -> str:
    try:
        clean = query.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "MultiAgentResearch/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            if extract:
                return f"Wikipedia: {data.get(\'title\', query)}\\n\\n{extract}"
        # Fallback: search API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query,
                  "format": "json", "srlimit": 3, "utf8": 1}
        resp = requests.get(search_url, params=params, timeout=10,
                            headers={"User-Agent": "MultiAgentResearch/1.0"})
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                top = results[0]["title"]
                r2 = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{top.replace(\' \', \'_\')}",
                    timeout=10, headers={"User-Agent": "MultiAgentResearch/1.0"})
                if r2.status_code == 200:
                    d = r2.json()
                    return f"Wikipedia: {d.get(\'title\', top)}\\n\\n{d.get(\'extract\', \'No extract.\')}"
        return f"Wikipedia: No information found for \'{query}\'."
    except Exception as e:
        return f"Wikipedia search error: {e}"

def search_duckduckgo(query: str) -> str:
    try:
        resp = requests.get("https://api.duckduckgo.com/",
                            params={"q": query, "format": "json",
                                    "no_redirect": "1", "no_html": "1", "skip_disambig": "1"},
                            timeout=10)
        if resp.status_code != 200:
            return "DuckDuckGo: No results."
        data = resp.json()
        parts = []
        if data.get("Abstract"):
            parts.append(f"Overview: {data[\'Abstract\']}")
        if data.get("Answer"):
            parts.append(f"Quick Answer: {data[\'Answer\']}")
        for t in data.get("RelatedTopics", [])[:3]:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(f"• {t[\'Text\'][:200]}")
        return "DuckDuckGo Results:\\n" + "\\n".join(parts) if parts else "DuckDuckGo: No instant answer available."
    except Exception as e:
        return f"DuckDuckGo error: {e}"

def gather_research(topic: str) -> str:
    wiki = search_wikipedia(topic)
    ddg = search_duckduckgo(topic)
    return f"=== RESEARCH DATA FOR: {topic} ===\\n\\n{wiki}\\n\\n---\\n\\n{ddg}\\n\\n=== END OF RESEARCH DATA ==="
'''

files["agents.py"] = '''"""
agents.py - Three specialized AI agents
Each agent = LLM + specific role + focused system prompt
"""
import requests, os
from dotenv import load_dotenv
from tools import gather_research

load_dotenv()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

def call_gemini(api_key, system_prompt, user_message, temperature=0.3, max_tokens=1500):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    resp = requests.post(GEMINI_URL, params={"key": api_key},
                         headers={"Content-Type": "application/json"},
                         json=payload, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

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
        print(f"  {self.name}: Analyzing with Gemini...")
        output = call_gemini(self.api_key, self.system_prompt,
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
        output = call_gemini(
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
        output = call_gemini(
            self.api_key, self.system_prompt,
            f\'Create a research summary about "{topic}":\\n\\nRESEARCH NOTES:\\n{verification_output["research_notes"]}\\n\\nVERIFICATION:\\n{verification_output["output"]}\',
            temperature=0.4, max_tokens=1500)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "output": output}
'''

files["orchestrator.py"] = '''"""
orchestrator.py - Coordinates all three agents in sequence
"""
import os
from dotenv import load_dotenv
from agents import ResearcherAgent, VerifierAgent, SummarizerAgent

load_dotenv()

class MultiAgentOrchestrator:
    """Manager that runs 3 agents in a pipeline: Research → Verify → Summarize"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required!")
        self.researcher = ResearcherAgent(self.api_key)
        self.verifier = VerifierAgent(self.api_key)
        self.summarizer = SummarizerAgent(self.api_key)
        print("✅ Multi-Agent Research System ready!")

    def research(self, topic, progress_callback=None):
        print(f"\\n🚀 Researching: \'{topic}\'\\n")
        results = {"topic": topic, "agents": [], "final_report": ""}
        try:
            # Step 1
            print("STEP 1/3: Researcher")
            r = self.researcher.run(topic)
            results["agents"].append({"name": self.researcher.name, "output": r["output"], "step": "Research"})
            if progress_callback: progress_callback(1, self.researcher.name, r["output"])
            # Step 2
            print("\\nSTEP 2/3: Verifier")
            v = self.verifier.run(r)
            results["agents"].append({"name": self.verifier.name, "output": v["output"], "step": "Verification"})
            if progress_callback: progress_callback(2, self.verifier.name, v["output"])
            # Step 3
            print("\\nSTEP 3/3: Summarizer")
            s = self.summarizer.run(v)
            results["agents"].append({"name": self.summarizer.name, "output": s["output"], "step": "Summary"})
            results["final_report"] = s["output"]
            if progress_callback: progress_callback(3, self.summarizer.name, s["output"])
            print("\\n✅ Research complete!")
            return results
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error: {e}")
            return results
'''

files["app.py"] = '''"""
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
    st.info("3 API calls per research session\\nFree tier: 20/day = 6 sessions/day")
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
                st.error(f"❌ Error: {results[\'error\']}")
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
                    with st.expander(f"{icons.get(a[\'step\'], \'🤖\')} {a[\'name\']} — {a[\'step\']}"):
                        st.markdown(a["output"])
        except Exception as e:
            st.error(f"❌ Error: {e}")
            if "429" in str(e):
                st.warning("⚠️ Daily quota reached. Wait until tomorrow or use a new API key.")
else:
    st.markdown("---")
    st.info("👆 Enter any topic above and click Research. The 3 agents will work together automatically.")
'''

files["requirements.txt"] = """streamlit>=1.28.0
requests>=2.28.0
python-dotenv>=1.0.0
"""

# Create the project folder and write all files
project_dir = "project2_multiagent"
os.makedirs(project_dir, exist_ok=True)

for filename, content in files.items():
    path = os.path.join(project_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created: {project_dir}/{filename}")

print(f"""
🎉 Project 2 created successfully!

Next steps:
1. cd {project_dir}
2. python -m venv venv
3. venv\\Scripts\\activate
4. pip install -r requirements.txt
5. streamlit run app.py

Then paste your Gemini API key and research any topic!
Uses only 3 API calls per research session (6 sessions/day free).
""")
