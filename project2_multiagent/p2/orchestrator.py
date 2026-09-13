"""
orchestrator.py - The Multi-Agent Orchestrator
================================================

The Orchestrator is the "manager" that coordinates all agents.
It decides:
    - Which agent runs first, second, third
    - How to pass information from one agent to the next
    - How to collect and return all results

Think of it like a relay race:
    Researcher → passes baton → Verifier → passes baton → Summarizer
"""

import os
from dotenv import load_dotenv
from agents import ResearcherAgent, VerifierAgent, SummarizerAgent

load_dotenv()


class MultiAgentOrchestrator:
    """
    Coordinates three specialized agents to research any topic.

    Pipeline:
        1. ResearcherAgent  — gathers + structures raw facts (1 API call)
        2. VerifierAgent    — checks accuracy + adds caveats (1 API call)
        3. SummarizerAgent  — writes polished final report (1 API call)

    Total: 3 API calls per research session.
    With 20 free calls/day → 6 full research sessions per day.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required! Set GEMINI_API_KEY in your .env file.")

        # Create all agents (they share the same API key)
        self.researcher = ResearcherAgent(self.api_key)
        self.verifier = VerifierAgent(self.api_key)
        self.summarizer = SummarizerAgent(self.api_key)

        print("✅ Multi-Agent Research System ready!")
        print(f"   Agents: {self.researcher.name}, {self.verifier.name}, {self.summarizer.name}")

    def research(self, topic: str, progress_callback=None) -> dict:
        """
        Run the full multi-agent research pipeline on a topic.

        Args:
            topic: The topic to research (e.g., "quantum computing")
            progress_callback: Optional function called after each agent
                               (used by Streamlit to update the UI live)

        Returns:
            dict with all agent outputs + final report
        """
        print(f"\n🚀 Starting multi-agent research on: '{topic}'\n")
        results = {"topic": topic, "agents": [], "final_report": ""}

        try:
            # ── STEP 1: Researcher Agent ──────────────────────────────────
            print("STEP 1/3: Researcher Agent")
            research_result = self.researcher.run(topic)
            results["agents"].append({
                "name": self.researcher.name,
                "output": research_result["output"],
                "step": "Research"
            })
            if progress_callback:
                progress_callback(1, self.researcher.name, research_result["output"])

            # ── STEP 2: Verifier Agent ────────────────────────────────────
            print("\nSTEP 2/3: Verifier Agent")
            verification_result = self.verifier.run(research_result)
            results["agents"].append({
                "name": self.verifier.name,
                "output": verification_result["output"],
                "step": "Verification"
            })
            if progress_callback:
                progress_callback(2, self.verifier.name, verification_result["output"])

            # ── STEP 3: Summarizer Agent ──────────────────────────────────
            print("\nSTEP 3/3: Summarizer Agent")
            summary_result = self.summarizer.run(verification_result)
            results["agents"].append({
                "name": self.summarizer.name,
                "output": summary_result["output"],
                "step": "Summary"
            })
            results["final_report"] = summary_result["output"]
            if progress_callback:
                progress_callback(3, self.summarizer.name, summary_result["output"])

            print(f"\n✅ Research complete! All 3 agents finished.\n")
            return results

        except Exception as e:
            error_msg = str(e)
            results["error"] = error_msg
            print(f"❌ Error during research: {error_msg}")
            return results
