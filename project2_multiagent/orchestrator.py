"""
orchestrator.py - Coordinates all three agents
"""
import os
from dotenv import load_dotenv
from agents import ResearcherAgent, VerifierAgent, SummarizerAgent, get_best_model

load_dotenv()


class MultiAgentOrchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required! Get a free one at console.groq.com")

        # Auto-detect the best available model
        self.model = get_best_model(self.api_key)

        self.researcher = ResearcherAgent(self.api_key, self.model)
        self.verifier = VerifierAgent(self.api_key, self.model)
        self.summarizer = SummarizerAgent(self.api_key, self.model)
        print(f"✅ Multi-Agent Research System ready! Model: {self.model}")

    def research(self, topic, progress_callback=None):
        print(f"\n🚀 Researching: '{topic}'\n")
        results = {"topic": topic, "agents": [], "final_report": ""}
        try:
            print("STEP 1/3: Researcher")
            r = self.researcher.run(topic)
            results["agents"].append({"name": self.researcher.name, "output": r["output"], "step": "Research"})
            if progress_callback:
                progress_callback(1, self.researcher.name, r["output"])

            print("\nSTEP 2/3: Verifier")
            v = self.verifier.run(r)
            results["agents"].append({"name": self.verifier.name, "output": v["output"], "step": "Verification"})
            if progress_callback:
                progress_callback(2, self.verifier.name, v["output"])

            print("\nSTEP 3/3: Summarizer")
            s = self.summarizer.run(v)
            results["agents"].append({"name": self.summarizer.name, "output": s["output"], "step": "Summary"})
            results["final_report"] = s["output"]
            if progress_callback:
                progress_callback(3, self.summarizer.name, s["output"])

            print("\n✅ Research complete!")
            return results
        except Exception as e:
            results["error"] = str(e)
            return results
