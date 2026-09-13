"""
agents.py - Three specialized AI agents using Groq API (FREE!)
Auto-detects available chat models (skips audio/vision models).
"""
import requests
import os
from dotenv import load_dotenv
from tools import gather_research

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models that are NOT chat models — skip these
SKIP_MODELS = ["whisper", "tts", "vision", "distil-whisper", "playai"]


def get_best_model(api_key):
    """Fetch available Groq models and pick the best chat model."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if resp.status_code == 200:
            all_models = [m["id"] for m in resp.json().get("data", [])]
            # Keep only chat/text models
            chat_models = [m for m in all_models
                           if not any(skip in m.lower() for skip in SKIP_MODELS)]
            print(f"  Chat models available: {chat_models}")

            preferred = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama3-70b-8192",
                "llama-3.1-8b-instant",
                "llama3-8b-8192",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ]
            for p in preferred:
                if p in chat_models:
                    print(f"  Using model: {p}")
                    return p
            if chat_models:
                print(f"  Using model: {chat_models[0]}")
                return chat_models[0]
    except Exception as e:
        print(f"  Could not fetch model list: {e}")
    return "gemma2-9b-it"


def call_groq(api_key, system_prompt, user_message, model, temperature=0.3, max_tokens=1500):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
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
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
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
        print(f"  {self.name}: Analyzing...")
        output = call_groq(self.api_key, self.system_prompt,
                           f'Analyze this research data about "{topic}":\n\n{raw_data}',
                           self.model, temperature=0.2, max_tokens=1200)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "raw_data": raw_data, "output": output}


class VerifierAgent:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
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
        output = call_groq(self.api_key, self.system_prompt,
                           f'Verify these notes about "{topic}":\n\n{research_output["output"]}\n\nSOURCE:\n{research_output["raw_data"][:2000]}',
                           self.model, temperature=0.1, max_tokens=1000)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic,
                "research_notes": research_output["output"], "output": output}


class SummarizerAgent:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
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
        output = call_groq(self.api_key, self.system_prompt,
                           f'Create a summary about "{topic}":\n\nNOTES:\n{verification_output["research_notes"]}\n\nVERIFICATION:\n{verification_output["output"]}',
                           self.model, temperature=0.4, max_tokens=1500)
        print(f"  {self.name}: Done ✓")
        return {"agent": self.name, "topic": topic, "output": output}
