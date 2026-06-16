"""
cts_bridge.py — CTS as the context management layer for the autoresearch harness.

Integrates the local CTS system (C:/Users/ASHMITH ATMURI/Documents/Codex/2026-04-29/cts)
into the Python research loop. Two jobs:

JOB 1 — TEXT CLEANING (smart_scrape)
  Port of cts-mcp/src/tools.ts `smartScrape`.
  Applied to eval stdout + experiment logs before compression and storage.
  Handles: repeated lines, stack traces, base64, long lines.
  No Node.js needed — pure Python.

JOB 2 — LLM WIKI BRIDGE (wiki_remember / wiki_recall)
  Port of cts-mcp/src/memory.ts `rememberSource` / `recallContext`.
  Writes research findings (skill files) into the CTS JSON wiki so they persist
  ACROSS sessions and are queryable from Claude Code via `cts_memory_recall`.
  This means: open a new Claude session, type "recall CMP research", and the
  last 50 experiments are there — not just in SQLite on a remote VM.

USAGE IN MEMORY.PY:
  from cts_bridge import smart_scrape, wiki_remember, wiki_recall
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── CTS wiki path — where cts-mcp/data/cts-memory/*.json files live ──────────
# Resolve relative to this file's location, or override with CTS_WIKI_DIR env var.
import os

_DEFAULT_CTS_ROOT = Path(os.environ.get(
    "CTS_ROOT",
    r"C:\Users\ASHMITH ATMURI\Documents\Codex\2026-04-29\cts"
))
_CTS_WIKI_DIR = _DEFAULT_CTS_ROOT / "cts-mcp" / "data" / "cts-memory"


# ═══════════════════════════════════════════════════════════════════════════════
# JOB 1 — SMART SCRAPE
# Port of cts-mcp/src/tools.ts smartScrape()
# Cleans raw text: repeated lines, stack traces, base64, long lines
# ═══════════════════════════════════════════════════════════════════════════════

def smart_scrape(text: str, max_line_length: int = 300) -> str:
    """
    Clean raw text output (eval stdout, session logs) before storing or compressing.
    Mirrors cts-mcp/src/tools.ts smartScrape() in Python.

    Steps:
      1. Multi-line deduplication (identical repeating blocks → single instance + count)
      2. Stack trace compression (keep first 3 + last 3 lines of deep traces)
      3. Base64 removal
      4. Long line truncation
    """
    lines = text.split("\n")

    # ── Step 1: multi-line deduplication ────────────────────────────────────
    deduped: list[str] = []
    i = 0
    while i < len(lines):
        matched = False
        for block_size in range(1, 6):
            if i + block_size > len(lines):
                break
            block = lines[i:i + block_size]

            # skip empty-only blocks
            if all(l.strip() == "" for l in block):
                break

            # count how many times this block repeats immediately
            repeat_count = 0
            j = i + block_size
            while j + block_size <= len(lines):
                next_block = lines[j:j + block_size]
                if block == next_block:
                    repeat_count += 1
                    j += block_size
                else:
                    break

            if repeat_count > 1:
                deduped.extend(block)
                deduped.append(f"... [{repeat_count} IDENTICAL BLOCKS COMPRESSED BY CTS] ...")
                i = j
                matched = True
                break

        if not matched:
            deduped.append(lines[i])
            i += 1

    # ── Step 2: stack trace compression ─────────────────────────────────────
    cleaned: list[str] = []
    in_stack = False
    stack_buffer: list[str] = []

    for line in deduped:
        is_stack = (
            line.strip().startswith("at ")
            or "Traceback " in line
            or "node_modules" in line
            or 'File "' in line
        )
        if is_stack:
            in_stack = True
            stack_buffer.append(line)
        else:
            if in_stack:
                if len(stack_buffer) > 6:
                    cleaned.extend(stack_buffer[:3])
                    cleaned.append(
                        f"... [{len(stack_buffer) - 6} STACK TRACE LINES COMPRESSED BY CTS] ..."
                    )
                    cleaned.extend(stack_buffer[-3:])
                else:
                    cleaned.extend(stack_buffer)
                stack_buffer = []
                in_stack = False
            cleaned.append(line)

    if in_stack:
        if len(stack_buffer) > 6:
            cleaned.extend(stack_buffer[:3])
            cleaned.append(
                f"... [{len(stack_buffer) - 6} STACK TRACE LINES COMPRESSED BY CTS] ..."
            )
            cleaned.extend(stack_buffer[-3:])
        else:
            cleaned.extend(stack_buffer)

    result = "\n".join(cleaned)

    # ── Step 3: base64 ───────────────────────────────────────────────────────
    result = re.sub(
        r"data:image/[a-zA-Z]*;base64,[A-Za-z0-9+/=]+",
        "[BASE64_IMAGE_REMOVED]",
        result
    )

    # ── Step 4: long line truncation ─────────────────────────────────────────
    final_lines = []
    for line in result.split("\n"):
        if len(line) > max_line_length:
            head = line[:150]
            tail = line[-50:]
            final_lines.append(
                f"{head} ... [LONG LINE OF {len(line)} CHARS TRUNCATED BY CTS] ... {tail}"
            )
        else:
            final_lines.append(line)

    return "\n".join(final_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# JOB 2 — LLM WIKI BRIDGE
# Port of cts-mcp/src/memory.ts rememberSource() / recallContext()
# Writes research skill files into the CTS JSON wiki for cross-session recall
# ═══════════════════════════════════════════════════════════════════════════════

def _wiki_path(session_id: str) -> Path:
    safe = re.sub(r"[^a-z0-9_.\-]+", "-", session_id.lower()).strip("-")[:80] or "codex"
    return _CTS_WIKI_DIR / f"{safe}.json"


def _load_wiki(session_id: str) -> dict:
    path = _wiki_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    now = datetime.now().isoformat()
    return {
        "sources": [],
        "pages": [],
        "indexMarkdown": "# Index\n\nNo pages yet.\n",
        "logMarkdown": "# Log\n",
        "schemaMarkdown": "# CTS LLM Wiki Schema\n\n- Sources: raw experiment records\n- Pages: synthesized wiki pages by concept\n",
        "version": 1,
        "lastUpdated": now,
    }


def _save_wiki(session_id: str, wiki: dict) -> None:
    path = _wiki_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wiki, indent=2) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "untitled"


def _summarize(content: str, max_chars: int = 900) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", content.replace("\s+", " ").strip())
    return " ".join(sentences[:8])[:max_chars] or content[:max_chars]


def _build_index(wiki: dict) -> str:
    lines = ["# Index", ""]
    for page in sorted(wiki["pages"], key=lambda p: p["path"]):
        lines.append(f"- [{page['path']}] {page['title']} — {', '.join(page.get('tags', []))}")
    return "\n".join(lines)


def _upsert_page(wiki: dict, page: dict) -> None:
    existing = next((p for p in wiki["pages"] if p["path"] == page["path"]), None)
    if existing is None:
        wiki["pages"].append(page)
    else:
        existing["markdown"] = f"{existing['markdown']}\n\n---\n\n{page['markdown']}"
        existing["sourceIds"] = list(set(existing.get("sourceIds", []) + page.get("sourceIds", [])))
        existing["tags"] = list(set(existing.get("tags", []) + page.get("tags", [])))
        existing["updatedAt"] = page["updatedAt"]


def wiki_remember(
    title: str,
    content: str,
    session_id: str = "cmp-research",
    tags: list[str] | None = None,
    outcome: str = "unknown",
    metric_delta: float = 0.0,
) -> str:
    """
    Write a research skill/finding into the CTS LLM Wiki.
    Mirrors cts-mcp/src/memory.ts rememberSource().

    Returns the path to the wiki JSON file.
    The wiki file can then be read by the CTS MCP tool in Claude Code sessions
    via cts_memory_recall (session_id="cmp-research", query="...").
    """
    wiki = _load_wiki(session_id)
    now = datetime.now().isoformat()
    source_id = f"src-{int(time.time() * 1000):x}"
    tags = list(set(["research", outcome] + (tags or [])))

    source: dict[str, Any] = {
        "id": source_id,
        "title": title[:120],
        "content": content.strip(),
        "addedAt": now,
        "outcome": outcome,
        "metric_delta": metric_delta,
    }
    wiki["sources"].append(source)

    # overview page (running log of all experiments)
    _upsert_page(wiki, {
        "path": "overview.md",
        "title": "Research Overview",
        "markdown": "\n".join([
            "# Research Overview",
            "",
            f"Latest experiment: {title}",
            "",
            "## Current synthesis",
            _summarize(content),
            "",
            "## Recent experiments",
            *[f"- {s['id']}: [{s.get('outcome','?')}] {s['title']}" for s in wiki["sources"][-8:]],
        ]),
        "updatedAt": now,
        "sourceIds": [source_id],
        "tags": tags,
    })

    # outcome-specific page
    outcome_path = f"outcomes/{outcome}.md"
    _upsert_page(wiki, {
        "path": outcome_path,
        "title": f"Outcomes: {outcome.upper()}",
        "markdown": "\n".join([
            f"# {outcome.upper()} Experiments",
            "",
            f"## {title}",
            f"*delta: {metric_delta:+.6f}*",
            "",
            _summarize(content, 600),
        ]),
        "updatedAt": now,
        "sourceIds": [source_id],
        "tags": tags,
    })

    # tag-based concept pages
    for tag in tags[:3]:
        tag_path = f"concepts/{_slug(tag)}.md"
        _upsert_page(wiki, {
            "path": tag_path,
            "title": tag.replace("-", " ").title(),
            "markdown": "\n".join([
                f"# {tag.replace('-', ' ').title()}",
                "",
                "## Notes",
                _summarize(content, 500),
                "",
                f"Sources: {source_id}",
            ]),
            "updatedAt": now,
            "sourceIds": [source_id],
            "tags": tags,
        })

    wiki["indexMarkdown"] = _build_index(wiki)
    wiki["logMarkdown"] += f"\n## [{now[:10]}] skill | {title}\n- id: {source_id} | outcome: {outcome} | delta: {metric_delta:+.4f}\n"
    wiki["version"] = wiki.get("version", 1) + 1
    wiki["lastUpdated"] = now

    _save_wiki(session_id, wiki)
    return str(_wiki_path(session_id))


def wiki_recall(
    query: str,
    session_id: str = "cmp-research",
    max_pages: int = 4,
) -> str:
    """
    Search the CTS LLM Wiki for relevant pages.
    Mirrors cts-mcp/src/memory.ts recallContext().

    Returns a formatted context block suitable for injection into agent prompts.
    This is the same data the CTS MCP tool returns when Ashmith types
    cts_memory_recall in Claude Code.
    """
    wiki = _load_wiki(session_id)
    if not wiki["pages"]:
        return f"CTS LLM Wiki ({session_id}): no pages yet."

    query_tokens = set(re.findall(r"\b[a-z][a-z0-9-]{2,}\b", query.lower()))
    scored = []
    for page in wiki["pages"]:
        haystack = f"{page['path']} {page['title']} {' '.join(page.get('tags', []))} {page['markdown']}".lower()
        tag_hits = sum(1 for t in page.get("tags", []) if query.lower() in t.lower()) * 3
        token_hits = sum(1 for t in query_tokens if t in haystack)
        score = tag_hits + token_hits
        if score > 0:
            scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:max_pages]]

    if not top:
        return f"CTS LLM Wiki ({session_id}): no pages matched '{query}'."

    lines = [f"CTS LLM Wiki context ({session_id}):"]
    for page in top:
        lines.append(f"\n## {page['path']}\n{page['markdown'][:1200]}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: available check
# ═══════════════════════════════════════════════════════════════════════════════

def cts_available() -> bool:
    """Returns True if the CTS wiki directory exists (CTS is installed locally)."""
    return _CTS_WIKI_DIR.parent.exists()


def cts_wiki_path(session_id: str = "cmp-research") -> Path:
    return _wiki_path(session_id)
