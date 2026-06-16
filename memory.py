"""
memory.py — Autoresearch Memory Layer

Three-level memory (Hermes pattern):
  SHORT-TERM   session.db messages table — WAL SQLite + FTS5 indexed
  MEDIUM-TERM  skills.db FTS5 index — searchable across past experiment skills
  LONG-TERM    wiki/skills/*.md — YAML+Markdown files, survive sessions/VM restarts

CTS integration (cts_bridge.py):
  CLEANING     smart_scrape() applied to eval output before storage and compression
  WIKI BRIDGE  skill_write() mirrors to CTS LLM Wiki JSON so findings are accessible
               from Claude Code sessions via cts_memory_recall("cmp-research", query)
  COMPRESSION  smart_scrape() cleans logs first, then three-tier Gemini compression
"""

from __future__ import annotations

import json
import re
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# CTS integration — context cleaning and cross-session wiki bridge
try:
    from cts_bridge import smart_scrape, wiki_remember, wiki_recall, cts_available
    _CTS_AVAILABLE = cts_available()
except ImportError:
    _CTS_AVAILABLE = False
    def smart_scrape(text: str, **kwargs) -> str: return text      # type: ignore
    def wiki_remember(*args, **kwargs) -> str: return ""           # type: ignore
    def wiki_recall(*args, **kwargs) -> str: return ""             # type: ignore

# ── paths (caller sets these before using) ────────────────────────────────────
WIKI_DIR   : Path | None = None
SKILLS_DIR : Path | None = None
SESSION_DB : Path | None = None
SKILLS_DB  : Path | None = None
SNAPSHOT   : Path | None = None   # .skills_prompt_snapshot.json (manifest cache)

GEMINI_MODEL_PLAIN = None   # set by loop_v2 after genai.configure()


def init(wiki_dir: Path, gemini_model):
    global WIKI_DIR, SKILLS_DIR, SESSION_DB, SKILLS_DB, SNAPSHOT, GEMINI_MODEL_PLAIN
    WIKI_DIR   = wiki_dir
    SKILLS_DIR = wiki_dir / "skills"
    SESSION_DB = wiki_dir / "session.db"
    SKILLS_DB  = wiki_dir / "skills.db"
    SNAPSHOT   = wiki_dir / ".skills_snapshot.json"
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    GEMINI_MODEL_PLAIN = gemini_model
    _setup_session_db()
    _setup_skills_db()


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY — Session SQLite (WAL mode, FTS5)
# mirrors hermes_state.py
# ═══════════════════════════════════════════════════════════════════════════════

def _session_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SESSION_DB))
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads + one writer (Hermes pattern)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _setup_session_db():
    conn = _session_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            started_at  REAL,
            ended_at    REAL,
            best_bpb    REAL,
            experiment_n INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            turn_n       INTEGER,
            role         TEXT,
            content      TEXT,
            timestamp    REAL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
            content, session_id, role,
            content='turns', content_rowid='id'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
            INSERT INTO turns_fts(rowid, content, session_id, role)
            VALUES (new.id, new.content, new.session_id, new.role);
        END
    """)
    conn.commit()
    conn.close()


_current_session_id: str | None = None
_turn_counter: int = 0


def session_start(session_id: str | None = None) -> str:
    global _current_session_id, _turn_counter
    _current_session_id = session_id or datetime.now().strftime("cmp_%Y%m%d_%H%M%S")
    _turn_counter = 0
    conn = _session_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at, best_bpb, experiment_n) VALUES (?,?,?,?)",
        (_current_session_id, datetime.now().timestamp(), 9.999, 0)
    )
    conn.commit()
    conn.close()
    return _current_session_id


def sync_turn(agent_name: str, content: str, response: str):
    """Record a turn (agent call + response) to short-term memory. Mirrors Hermes sync_all()."""
    global _turn_counter
    if _current_session_id is None:
        return
    conn = _session_conn()
    _turn_counter += 1
    ts = datetime.now().timestamp()
    conn.execute(
        "INSERT INTO turns(session_id, turn_n, role, content, timestamp) VALUES (?,?,?,?,?)",
        (_current_session_id, _turn_counter, f"prompt:{agent_name}", content[:4000], ts)
    )
    conn.execute(
        "INSERT INTO turns(session_id, turn_n, role, content, timestamp) VALUES (?,?,?,?,?)",
        (_current_session_id, _turn_counter, f"response:{agent_name}", response[:4000], ts)
    )
    conn.commit()
    conn.close()


def session_fts_search(query: str, k: int = 5) -> list[dict]:
    """FTS5 search across ALL past session turns. Hermes: messages_fts."""
    conn = _session_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM turns_fts WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, k)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM-TERM MEMORY — Skills SQLite FTS5 index
# ═══════════════════════════════════════════════════════════════════════════════

def _skills_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SKILLS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _setup_skills_db():
    conn = _skills_conn()
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS skills USING fts5(
            name, tags, outcome, bpb_delta, content
        )
    """)
    conn.commit()
    conn.close()


def skill_search(query: str, k: int = 5) -> list[dict]:
    conn = _skills_conn()
    try:
        rows = conn.execute(
            "SELECT name, outcome, bpb_delta, content FROM skills "
            "WHERE skills MATCH ? ORDER BY rank LIMIT ?",
            (query, k)
        ).fetchall()
        return [{"name": r[0], "outcome": r[1], "bpb_delta": r[2], "content": r[3]} for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY — Skill files (YAML+Markdown)
# mirrors hermes: skill_utils.py + prompt_builder.py manifest cache
# ═══════════════════════════════════════════════════════════════════════════════

def skill_write(name: str, tags: list[str], outcome: str,
                bpb_delta: float, content: str) -> Path:
    """Write skill file + index in SQLite. Mirrors Hermes skill lifecycle."""
    safe = re.sub(r"[^\w\-]", "_", name)[:60]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SKILLS_DIR / f"{ts}_{safe}.md"

    yaml = textwrap.dedent(f"""\
        ---
        skill: {safe}
        created: {datetime.now().isoformat()}
        outcome: {outcome}
        bpb_delta: {bpb_delta:+.6f}
        tags: [{", ".join(tags)}]
        ---
    """)
    path.write_text(yaml + "\n" + content)

    conn = _skills_conn()
    conn.execute(
        "INSERT INTO skills(name, tags, outcome, bpb_delta, content) VALUES (?,?,?,?,?)",
        (safe, " ".join(tags), outcome, str(bpb_delta), content)
    )
    conn.commit()
    conn.close()

    _invalidate_snapshot()

    # Mirror to CTS LLM Wiki — makes findings accessible via cts_memory_recall
    # in Claude Code sessions ("what did the CMP harness learn last night?")
    if _CTS_AVAILABLE:
        try:
            wiki_remember(
                title=safe,
                content=content,
                session_id="cmp-research",
                tags=tags,
                outcome=outcome,
                metric_delta=bpb_delta,
            )
        except Exception:
            pass  # wiki bridge failure never blocks research

    return path


def build_skills_context_block(query: str = "", k: int = 5) -> str:
    """
    Build a skills context block for Planner's system prompt.
    Uses manifest cache (Hermes: .skills_prompt_snapshot.json) — avoids rescanning every turn.
    """
    snapshot = _load_snapshot()
    current_manifest = _build_manifest()

    if snapshot.get("manifest") == current_manifest and snapshot.get("query") == query:
        return snapshot.get("block", "")

    # manifest changed or query changed — rebuild
    skills = skill_search(query, k=k) if query else _load_all_skills(k)
    if not skills:
        block = "(no skill files yet — first session)"
    else:
        lines = []
        for s in skills:
            lines.append(
                f"[{s['outcome'].upper()}] {s['name']} (delta={s['bpb_delta']}) — {s['content'][:300]}"
            )
        block = "\n\n".join(lines)

    # cache it
    _save_snapshot({"manifest": current_manifest, "query": query, "block": block})
    return block


def _build_manifest() -> dict[str, list]:
    """Map skill filename → [mtime_ns, size]. Hermes: skills manifest cache."""
    m = {}
    if SKILLS_DIR and SKILLS_DIR.exists():
        for f in sorted(SKILLS_DIR.glob("*.md")):
            stat = f.stat()
            m[f.name] = [stat.st_mtime_ns, stat.st_size]
    return m


def _load_snapshot() -> dict:
    if SNAPSHOT and SNAPSHOT.exists():
        try:
            return json.loads(SNAPSHOT.read_text())
        except Exception:
            pass
    return {}


def _save_snapshot(data: dict):
    if SNAPSHOT:
        SNAPSHOT.write_text(json.dumps(data))


def _invalidate_snapshot():
    if SNAPSHOT and SNAPSHOT.exists():
        SNAPSHOT.unlink()


def _load_all_skills(k: int) -> list[dict]:
    if not SKILLS_DIR:
        return []
    files = sorted(SKILLS_DIR.glob("*.md"), reverse=True)[:k]
    result = []
    for f in files:
        content = f.read_text()
        parts = content.split("---", 2)
        body = parts[2].strip() if len(parts) >= 3 else content
        name = f.stem
        outcome = "keep" if "outcome: keep" in content else "discard"
        bpb_delta = "?"
        m = re.search(r"bpb_delta: ([+-]?\d+\.\d+)", content)
        if m:
            bpb_delta = m.group(1)
        result.append({"name": name, "outcome": outcome, "bpb_delta": bpb_delta, "content": body[:400]})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT COMPRESSION — three-tier (Hermes: context_compressor.py)
# Protect head (system + first 3) + tail (last 6 messages worth)
# Compress middle with cheaper Gemini call
# ═══════════════════════════════════════════════════════════════════════════════

def compress_context(text: str, max_chars: int = 5000) -> str:
    """
    Two-stage compression:
      STAGE 1 (CTS smart_scrape): clean repeated lines, stack traces, long lines
      STAGE 2 (Hermes three-tier): HEAD + TAIL always kept, MIDDLE compressed with Gemini

    If text fits in max_chars after stage 1, returns without calling Gemini.
    """
    # Stage 1: CTS text cleaning — handles repeated progress lines, crash traces, etc.
    if _CTS_AVAILABLE:
        text = smart_scrape(text)

    if len(text) <= max_chars:
        return text

    head = text[:800]
    tail = text[-1200:]
    middle = text[800:-1200]

    if not middle.strip():
        return head + tail

    if GEMINI_MODEL_PLAIN is None:
        # no model yet — just truncate middle
        middle_compressed = middle[:600] + "\n...[compressed]..."
    else:
        try:
            prompt = f"""Compress this experiment log section to under 400 words.
Preserve: val_bpb numbers, keep/discard decisions, failure reasons, key learnings.
Drop: repetitive progress lines, exact code snippets, redundant timestamps.

LOG SECTION:
{middle[:8000]}
"""
            resp = GEMINI_MODEL_PLAIN.generate_content(prompt)
            middle_compressed = resp.text.strip()
            saved = len(middle) - len(middle_compressed)
            pct   = int(100 * saved / max(len(middle), 1))
        except Exception as e:
            middle_compressed = middle[:600] + f"\n...[compression failed: {e}]..."
            saved, pct = 0, 0

    result = head + "\n\n[...COMPRESSED MIDDLE...]\n" + middle_compressed + "\n[...END COMPRESSED...]\n\n" + tail
    return result


def prefetch(query: str) -> str:
    """
    Build a memory context block for the Planner's system prompt.
    Three sources:
      1. FTS5 short-term session turns (this session)
      2. Manifest-cached skill block (all past sessions on disk)
      3. CTS LLM Wiki (cross-session recall — survives VM restarts, accessible from Claude Code)
    """
    session_hits = session_fts_search(query, k=3)
    skill_block  = build_skills_context_block(query=query, k=5)

    session_ctx = ""
    if session_hits:
        session_ctx = "### Recent session context (FTS match)\n" + "\n".join(
            f"- [{h['role']}]: {h['content'][:200]}" for h in session_hits
        )

    # CTS wiki recall — finds relevant past findings by concept
    cts_ctx = ""
    if _CTS_AVAILABLE:
        try:
            cts_result = wiki_recall(query, session_id="cmp-research", max_pages=2)
            if "no pages yet" not in cts_result and "no pages matched" not in cts_result:
                cts_ctx = f"\n\n### CTS Wiki (cross-session)\n{cts_result[:800]}"
        except Exception:
            pass

    return f"{session_ctx}\n\n### Skill memory\n{skill_block}{cts_ctx}".strip()
