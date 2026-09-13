"""
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
        print(f"  {self.name}: Gathering data about '{topic}'...")
        raw_data = gather_research(topic)
        print(f"  {self.name}: Analyzing with Gemini...")
        output = call_gemini(self.api_key, self.system_prompt,
                             f'Please analyze this research data about "{topic}":\n\n{raw_data}',
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
            f'Verify these research notes about "{topic}":\n\n{research_output["output"]}\n\nORIGINAL DATA:\n{research_output["raw_data"][:2000]}',
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
            f'Create a research summary about "{topic}":\n\nRESEARCH NOTES:\n{verification_output["research_notes"]}\n\nVERIFICATION:\n{verification_output["output"]}',
            temperature=0.4, max_tokens=1500)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "output": output}
