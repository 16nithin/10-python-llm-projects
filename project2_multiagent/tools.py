"""
tools.py - Search tools for agents (FREE - no API key needed!)
Wikipedia API + DuckDuckGo Instant Answers
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
                return f"Wikipedia: {data.get('title', query)}\n\n{extract}"

        # Fallback: use Wikipedia search API
        params = {"action": "query", "list": "search", "srsearch": query,
                  "format": "json", "srlimit": 3, "utf8": 1}
        resp = requests.get("https://en.wikipedia.org/w/api.php", params=params,
                            timeout=10, headers={"User-Agent": "MultiAgentResearch/1.0"})
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                top = results[0]["title"]
                r2 = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{top.replace(' ', '_')}",
                    timeout=10, headers={"User-Agent": "MultiAgentResearch/1.0"})
                if r2.status_code == 200:
                    d = r2.json()
                    return f"Wikipedia: {d.get('title', top)}\n\n{d.get('extract', 'No extract.')}"

        return f"Wikipedia: No information found for '{query}'."
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
            parts.append(f"Overview: {data['Abstract']}")
        if data.get("Answer"):
            parts.append(f"Quick Answer: {data['Answer']}")
        for t in data.get("RelatedTopics", [])[:3]:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(f"• {t['Text'][:200]}")
        return "DuckDuckGo Results:\n" + "\n".join(parts) if parts else "DuckDuckGo: No instant answer available."
    except Exception as e:
        return f"DuckDuckGo error: {e}"


def gather_research(topic: str) -> str:
    wiki = search_wikipedia(topic)
    ddg = search_duckduckgo(topic)
    return f"=== RESEARCH DATA FOR: {topic} ===\n\n{wiki}\n\n---\n\n{ddg}\n\n=== END OF RESEARCH DATA ==="
