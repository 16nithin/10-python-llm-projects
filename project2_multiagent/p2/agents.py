"""
agents.py - The three specialized AI agents
============================================

What is an Agent?
    An agent = LLM + a specific role + tools + a goal.
    Instead of one LLM doing everything, we give each agent
    ONE job it's really good at. Like a team of specialists.

Our three agents:
    1. ResearcherAgent  — finds and collects raw information
    2. VerifierAgent    — checks facts, flags uncertain claims
    3. SummarizerAgent  — writes the final clean report

Each agent calls Gemini with a specialized system prompt
that tells it exactly what role to play.
"""

import requests
import os
from dotenv import load_dotenv
from tools import gather_research

load_dotenv()

# Gemini REST API endpoint
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"


def call_gemini(api_key: str, system_prompt: str, user_message: str,
                temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """
    Call the Gemini API with a system prompt and user message.
    This is the core function all agents use to "think".
    """
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_message}]}
        ],
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )

    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")

    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: The Researcher
# ─────────────────────────────────────────────────────────────────────────────

class ResearcherAgent:
    """
    The Researcher gathers raw information about a topic.

    It uses real tools (Wikipedia, DuckDuckGo) to collect facts,
    then uses Gemini to organize those facts into structured notes.

    Output → passed to VerifierAgent
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "🔍 Researcher Agent"

        self.system_prompt = """You are a Research Specialist AI agent. Your job is to:

1. Analyze the raw research data provided to you
2. Extract the most important facts and information
3. Organize them into clear, structured research notes
4. Include specific details: dates, numbers, names, events
5. Note any gaps or areas where information is limited

Format your output as:
## KEY FACTS
(bullet points of the most important facts)

## BACKGROUND CONTEXT
(1-2 paragraphs of context)

## IMPORTANT DETAILS
(specific dates, numbers, names, statistics)

## GAPS IN RESEARCH
(what information is missing or unclear)

Be thorough and accurate. Only include information from the provided data."""

    def run(self, topic: str) -> dict:
        """
        Research a topic: gather data from tools, then analyze with Gemini.
        Returns dict with 'output' (text) and 'raw_data' (from tools).
        """
        print(f"  {self.name}: Searching for information about '{topic}'...")

        # Step 1: Use tools to gather raw data (NO API call)
        raw_data = gather_research(topic)
        print(f"  {self.name}: Data gathered. Now analyzing...")

        # Step 2: Use Gemini to analyze and structure the data (1 API call)
        user_message = f"""Please analyze this research data about "{topic}" and extract key information:

{raw_data}"""

        output = call_gemini(
            self.api_key,
            self.system_prompt,
            user_message,
            temperature=0.2,
            max_tokens=1200
        )

        print(f"  {self.name}: Research complete ✓")
        return {
            "agent": self.name,
            "topic": topic,
            "raw_data": raw_data,
            "output": output
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: The Verifier
# ─────────────────────────────────────────────────────────────────────────────

class VerifierAgent:
    """
    The Verifier checks the Researcher's output for accuracy.

    It looks for:
    - Contradictions or inconsistencies
    - Claims that seem uncertain or need caveats
    - Information that is well-supported vs. questionable
    - Overall reliability assessment

    Input  ← ResearcherAgent output
    Output → passed to SummarizerAgent
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "✅ Verifier Agent"

        self.system_prompt = """You are a Fact-Checking and Verification Specialist AI agent. Your job is to:

1. Review research notes critically
2. Identify claims that are well-supported vs. uncertain
3. Spot any contradictions or inconsistencies
4. Add appropriate caveats to uncertain claims
5. Rate the overall reliability of the research

Format your output as:
## VERIFIED FACTS ✓
(facts that appear reliable and well-supported)

## UNCERTAIN CLAIMS ⚠️
(claims that need caveats or are potentially inaccurate)

## CONTRADICTIONS OR ISSUES ❌
(any contradictions found, or "None found" if clean)

## RELIABILITY ASSESSMENT
(1-2 sentences rating the overall quality of this research, score it 1-10)

## RECOMMENDED CAVEATS
(what should readers know about the limitations of this information)

Be critical but fair. Your goal is accuracy, not negativity."""

    def run(self, research_output: dict) -> dict:
        """
        Verify the researcher's findings.
        Input: the output dict from ResearcherAgent
        """
        print(f"  {self.name}: Verifying research findings...")

        topic = research_output["topic"]
        research_notes = research_output["output"]
        raw_data = research_output["raw_data"]

        user_message = f"""Please verify these research notes about "{topic}".

RESEARCH NOTES TO VERIFY:
{research_notes}

ORIGINAL SOURCE DATA (for cross-referencing):
{raw_data[:2000]}"""

        output = call_gemini(
            self.api_key,
            self.system_prompt,
            user_message,
            temperature=0.1,  # Low temperature = more consistent, careful
            max_tokens=1000
        )

        print(f"  {self.name}: Verification complete ✓")
        return {
            "agent": self.name,
            "topic": topic,
            "research_notes": research_notes,
            "output": output
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: The Summarizer
# ─────────────────────────────────────────────────────────────────────────────

class SummarizerAgent:
    """
    The Summarizer creates the final polished report.

    It combines the Researcher's facts and the Verifier's
    quality assessment into one clean, readable document
    that a human would actually want to read.

    Input  ← VerifierAgent output (which includes ResearcherAgent output)
    Output → Final report shown to user
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "📝 Summarizer Agent"

        self.system_prompt = """You are a Professional Research Writer AI agent. Your job is to:

1. Take verified research notes and create a polished summary
2. Write in clear, engaging language anyone can understand
3. Structure information logically
4. Include important caveats from the verification stage
5. Make it genuinely useful to someone who wants to learn about this topic

Format your output as:
# [Topic Name]: Research Summary

## Overview
(2-3 sentence introduction)

## Key Findings
(3-5 bullet points of the most important facts)

## Detailed Analysis
(2-3 paragraphs with depth and context)

## Reliability Notes
(brief note about confidence level and any caveats)

## Bottom Line
(1-2 sentences: what's the most important thing to know?)

Write for an intelligent general audience. Be informative and engaging."""

    def run(self, verification_output: dict) -> dict:
        """
        Create the final summary report.
        Input: the output dict from VerifierAgent
        """
        print(f"  {self.name}: Writing final summary...")

        topic = verification_output["topic"]
        research_notes = verification_output["research_notes"]
        verification_notes = verification_output["output"]

        user_message = f"""Create a polished research summary about "{topic}" using these materials:

RESEARCH NOTES:
{research_notes}

VERIFICATION ANALYSIS:
{verification_notes}"""

        output = call_gemini(
            self.api_key,
            self.system_prompt,
            user_message,
            temperature=0.4,  # Slightly higher = more engaging writing
            max_tokens=1500
        )

        print(f"  {self.name}: Summary complete ✓")
        return {
            "agent": self.name,
            "topic": topic,
            "output": output
        }
