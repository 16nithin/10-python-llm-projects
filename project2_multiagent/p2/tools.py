"""
tools.py - Search tools for agents
These are FREE — no API key needed!
- Wikipedia API: returns encyclopedia summaries
- DuckDuckGo Instant Answers: returns quick facts
"""

import requests


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    Uses Wikipedia's free REST API — no key needed.

    Returns a text summary of the topic.
    """
    try:
        # First: try direct page lookup
        clean_query = query.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "MultiAgentResearch/1.0"})

        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", query)
            extract = data.get("extract", "")
            if extract:
                return f"Wikipedia: {title}\n\n{extract}"

        # Second: use Wikipedia search to find the right page
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
            "utf8": 1
        }
        resp = requests.get(search_url, params=params, timeout=10,
                            headers={"User-Agent": "MultiAgentResearch/1.0"})

        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                top_title = results[0]["title"]
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_title.replace(' ', '_')}"
                resp2 = requests.get(summary_url, timeout=10,
                                     headers={"User-Agent": "MultiAgentResearch/1.0"})
                if resp2.status_code == 200:
                    data = resp2.json()
                    return f"Wikipedia: {data.get('title', top_title)}\n\n{data.get('extract', 'No extract available.')}"

        return f"Wikipedia: No information found for '{query}'."

    except Exception as e:
        return f"Wikipedia search error: {e}"


def search_duckduckgo(query: str) -> str:
    """
    Use DuckDuckGo Instant Answers API for quick facts.
    Completely free, no API key needed.

    Returns instant answer text if available.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1"
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            return "DuckDuckGo: No results."

        data = resp.json()
        parts = []

        if data.get("Abstract"):
            parts.append(f"Overview: {data['Abstract']}")

        if data.get("Answer"):
            parts.append(f"Quick Answer: {data['Answer']}")

        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"• {topic['Text'][:200]}")

        if parts:
            return "DuckDuckGo Results:\n" + "\n".join(parts)

        return "DuckDuckGo: No instant answer available for this topic."

    except Exception as e:
        return f"DuckDuckGo search error: {e}"


def gather_research(topic: str) -> str:
    """
    Gather research on a topic using all available tools.
    Combines Wikipedia + DuckDuckGo results.
    """
    wiki_result = search_wikipedia(topic)
    ddg_result = search_duckduckgo(topic)

    combined = f"""=== RESEARCH DATA FOR: {topic} ===

{wiki_result}

---

{ddg_result}

=== END OF RESEARCH DATA ==="""

    return combined
