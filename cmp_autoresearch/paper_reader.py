"""
paper_reader.py — Literature Agent for ARIA v2.
Fetches papers from Semantic Scholar + arXiv. Returns knowledge cards.
No API key required — both have free public APIs (rate-limited).
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class KnowledgeCard:
    title: str
    authors: str
    year: int
    abstract_snippet: str
    source: str   # "semantic_scholar" or "arxiv"
    url: str


def _semantic_scholar(query: str, limit: int = 3) -> list[KnowledgeCard]:
    try:
        q = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={q}&limit={limit}&fields=title,authors,year,abstract,url"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ARIA-Research/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        cards = []
        for p in data.get("data", []):
            abstract = (p.get("abstract") or "").strip()
            if not abstract:
                continue
            cards.append(KnowledgeCard(
                title=p.get("title", "").strip(),
                authors=", ".join(a.get("name", "") for a in p.get("authors", [])[:3]),
                year=p.get("year") or 2024,
                abstract_snippet=abstract[:300],
                source="semantic_scholar",
                url=p.get("url", ""),
            ))
        return cards
    except Exception as e:
        print(f"  [lit] Semantic Scholar error: {e}")
        return []


def _arxiv(query: str, limit: int = 3) -> list[KnowledgeCard]:
    try:
        q = urllib.parse.quote(query)
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=all:{q}&max_results={limit}&sortBy=relevance"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ARIA-Research/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
        cards = []
        for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.DOTALL)[:limit]:
            def _tag(tag):
                m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
                return m.group(1).strip() if m else ""
            title = _tag("title")
            abstract = _tag("summary")
            url_val = _tag("id")
            authors = ", ".join(re.findall(r"<name>(.*?)</name>", entry)[:3])
            year_m = re.search(r"<published>(\d{4})", entry)
            year = int(year_m.group(1)) if year_m else 2024
            if not abstract:
                continue
            cards.append(KnowledgeCard(
                title=title,
                authors=authors,
                year=year,
                abstract_snippet=abstract[:300],
                source="arxiv",
                url=url_val,
            ))
        return cards
    except Exception as e:
        print(f"  [lit] arXiv error: {e}")
        return []


def search_papers(query: str, max_papers: int = 4) -> list[KnowledgeCard]:
    """
    Search Semantic Scholar + arXiv for a query. Returns knowledge cards.
    Called before each council run to ground hypotheses in literature.
    Fails gracefully — loop continues even if both APIs are down.
    """
    if not query or not query.strip():
        return []

    print(f"  [lit] Searching: '{query}'")

    half = max(2, max_papers // 2)
    cards = _semantic_scholar(query, limit=half)
    time.sleep(0.4)   # respect rate limit
    remaining = max(0, max_papers - len(cards))
    if remaining:
        cards += _arxiv(query, limit=remaining)

    print(f"  [lit] {len(cards)} papers found")
    return cards[:max_papers]


def format_for_council(cards: list[KnowledgeCard]) -> str:
    """Format knowledge cards as a section injected into the council prompt."""
    if not cards:
        return ""
    lines = ["\n## Literature Context (cite with [N] in your hypothesis)"]
    for i, c in enumerate(cards, 1):
        lines.append(
            f"\n[{i}] **{c.title}** ({c.year}) — {c.authors}"
            f"\n    {c.abstract_snippet[:250]}"
            f"\n    Source: {c.source} | {c.url}"
        )
    lines.append(
        "\nYour hypothesis MUST cite at least one paper above with [N] "
        "if the paper is directly relevant."
    )
    return "\n".join(lines)
