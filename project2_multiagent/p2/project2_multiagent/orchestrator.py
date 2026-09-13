"""
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
        print(f"\n🚀 Researching: '{topic}'\n")
        results = {"topic": topic, "agents": [], "final_report": ""}
        try:
            # Step 1
            print("STEP 1/3: Researcher")
            r = self.researcher.run(topic)
            results["agents"].append({"name": self.researcher.name, "output": r["output"], "step": "Research"})
            if progress_callback: progress_callback(1, self.researcher.name, r["output"])
            # Step 2
            print("\nSTEP 2/3: Verifier")
            v = self.verifier.run(r)
            results["agents"].append({"name": self.verifier.name, "output": v["output"], "step": "Verification"})
            if progress_callback: progress_callback(2, self.verifier.name, v["output"])
            # Step 3
            print("\nSTEP 3/3: Summarizer")
            s = self.summarizer.run(v)
            results["agents"].append({"name": self.summarizer.name, "output": s["output"], "step": "Summary"})
            results["final_report"] = s["output"]
            if progress_callback: progress_callback(3, self.summarizer.name, s["output"])
            print("\n✅ Research complete!")
            return results
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error: {e}")
            return results
