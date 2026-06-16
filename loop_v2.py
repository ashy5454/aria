"""
loop_v2.py — Autoresearch Loop v2 (Hermes Agent Pattern)

Generic ML architecture research harness. Configured via research.yaml.
Point it at any model file + eval script and it runs experiments autonomously,
keeps what works, discards what doesn't, and STOPS when it proves something.

Ported from NousResearch/hermes-agent cognitive architecture:
  MEMORY     (memory.py): WAL SQLite+FTS5 session / FTS5 skill index / Markdown skill files
  FOUR AGENTS per experiment:
    1. Planner   — reads memory + web → proposes hypothesis
    2. Coder     — implements ONE change in the model file
    3. Evaluator — calls eval script (immutable), parses metric
    4. Analyst   — interprets result, writes skill file, keep/discard
  GEPA-LITE: every EVOLVE_EVERY experiments, Evolver rewrites AGENT.md
  COUNCIL: karpathy/llm-council — 4 models propose + rank + chairman synthesizes
           (requires OPENROUTER_API_KEY; falls back to single Gemini otherwise)

STOP CONDITIONS (writes CONCLUSION.md + exits cleanly):
  1. metric crosses goal (e.g. val_bpb < 1.91)
  2. N consecutive experiments with no improvement
  3. Same mechanism confirmed in N independent runs

Usage:
    export GEMINI_API_KEY=your_key_here
    python loop_v2.py                          # uses research.yaml in same dir
    python loop_v2.py --config my_project.yaml # use different config
    tmux new -s research   # run in tmux so you can detach
    Ctrl+B D to detach, tmux attach -t research to reattach
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from providers import make_client
import memory as mem

try:
    import yaml as _yaml
    def _load_yaml(path: Path) -> dict:
        return _yaml.safe_load(path.read_text()) if path.exists() else {}
except ImportError:
    import json as _json
    def _load_yaml(path: Path) -> dict:
        return {}   # yaml not installed; use defaults


# ── config ────────────────────────────────────────────────────────────────────

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--config", default=None)
_args, _ = _ap.parse_known_args()

_cfg_path = (
    Path(_args.config) if _args.config else Path(__file__).parent / "research.yaml"
)
CFG = _load_yaml(_cfg_path)
_cfg_dir = _cfg_path.parent   # paths in config are relative to the config file

_loop    = CFG.get("loop", {})
_eval    = CFG.get("eval", {})
_domain  = CFG.get("domain", {})
_stop    = CFG.get("stop_conditions", {})
_proj    = CFG.get("project", {})
_model   = CFG.get("model", {})

# ── tunable constants (from config, with CMP-safe defaults) ──────────────────

PROVIDER        = _loop.get("provider", "gemini")
GEMINI_MODEL    = _loop.get("gemini_model", _loop.get("model", "gemini-3.5-flash"))
EVOLVE_EVERY    = _loop.get("evolve_every", 20)
SKILLS_SEARCH_K = _loop.get("skills_search_k", 5)
MAX_RETRIES     = _loop.get("max_retries", 3)
USE_WEB_SEARCH  = _loop.get("use_web_search", True)

METRIC_NAME      = _eval.get("metric", "val_bpb")
METRIC_KEY       = _eval.get("metric_output_key", "eval")   # key in stdout: "eval=X.XXXX"
METRIC_DIRECTION = _eval.get("direction", "lower")          # "lower" or "higher"
METRIC_GOAL      = float(_eval.get("goal", 1.91))
EVAL_ARGS        = _eval.get("args", ["--pc", "off"])       # extra args to eval script

DOMAIN_CONTEXT   = _domain.get("context", "")
PROJECT_NAME     = _proj.get("name", "ML Architecture Research")

CONSECUTIVE_FAIL_LIMIT = int(_stop.get("consecutive_failures", 20))
MECHANISM_CONFIRM_N    = int(_stop.get("mechanism_confirmed", 3))

# Multi-persona Gemini council — always active (uses GEMINI_API_KEY, no OpenRouter needed)
_COUNCIL_AVAILABLE = True
from hypothesis_council import run_hypothesis_council, run_result_interpretation

# ── paths ─────────────────────────────────────────────────────────────────────

HARNESS_DIR  = Path(__file__).parent
ROOT         = HARNESS_DIR.parent
SOLUTION     = HARNESS_DIR / _model.get("file", "solutions/cmp_current.py")
EVAL_SCRIPT  = HARNESS_DIR / _eval.get("script", "harness/evaluate.py")
RESULTS_DIR  = HARNESS_DIR / "results"
WIKI_DIR     = HARNESS_DIR / "wiki"
AGENT_MD     = HARNESS_DIR / "AGENT.md"
RESULTS_TSV  = RESULTS_DIR / "results.tsv"
SESSION_LOG  = RESULTS_DIR / "session.log"
CONCLUSION   = RESULTS_DIR / "CONCLUSION.md"

for d in [RESULTS_DIR, WIKI_DIR]:
    d.mkdir(exist_ok=True)

# ── LLM provider setup ────────────────────────────────────────────────────────

try:
    _llm         = make_client(PROVIDER, GEMINI_MODEL, use_search=USE_WEB_SEARCH)
    model_plain  = _llm   # kept for mem.init() which calls model_plain.generate_content
    model_search = _llm
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── boot memory layer (Hermes: WAL SQLite + FTS5 + manifest cache) ────────────
mem.init(WIKI_DIR, model_plain)
mem.session_start()


def load_recent_results(n: int = 15) -> str:
    if not RESULTS_TSV.exists():
        return "(no experiments yet)"
    lines = RESULTS_TSV.read_text().strip().splitlines()
    if len(lines) < 2:
        return "(no experiments yet)"
    return "\n".join(lines[-n:])


# ── helpers: is this an improvement? ─────────────────────────────────────────

def is_improvement(new_val: float, best_val: float) -> bool:
    if METRIC_DIRECTION == "lower":
        return new_val < best_val
    return new_val > best_val


def beats_goal(val: float) -> bool:
    if METRIC_DIRECTION == "lower":
        return val < METRIC_GOAL
    return val > METRIC_GOAL


def worst_possible() -> float:
    return 9.999999 if METRIC_DIRECTION == "lower" else -9.999999


# ═══════════════════════════════════════════════════════════════════════════════
# GIT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def git(cmd: str) -> str:
    r = subprocess.run(f"git {cmd}", shell=True, capture_output=True, text=True, cwd=str(ROOT))
    return r.stdout.strip()


def git_commit(message: str) -> str:
    rel_solution = SOLUTION.relative_to(ROOT)
    subprocess.run(f"git add {rel_solution}", shell=True, cwd=str(ROOT), capture_output=True)
    subprocess.run(["git", "-c", "user.email=team@yudi.co.in", "-c", "user.name=ashy5454",
                    "commit", "-m", message],
                   cwd=str(ROOT), capture_output=True)
    return git("rev-parse --short HEAD")


def git_reset() -> None:
    r = subprocess.run("git reset --hard HEAD~1", shell=True, cwd=str(ROOT), capture_output=True)
    if r.returncode != 0:
        log(f"[git] reset failed: {r.stderr.decode()[:100]} — forcing checkout instead")
        subprocess.run(f"git checkout -- {SOLUTION.relative_to(ROOT)}",
                       shell=True, cwd=str(ROOT), capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(SESSION_LOG, "a") as f:
        f.write(line + "\n")


def append_result(commit: str, metric_val: float, params_m: float,
                  status: str, description: str):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(f"commit\t{METRIC_NAME}\tparams_M\tstatus\tdescription\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{metric_val:.6f}\t{params_m:.2f}\t{status}\t{description}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CONCLUSION WRITER — called when any stop condition is met
# ═══════════════════════════════════════════════════════════════════════════════

def write_conclusion(reason: str, best_metric: float, experiment_n: int,
                     keeps: int, discards: int):
    lines_tsv = RESULTS_TSV.read_text().strip().splitlines() if RESULTS_TSV.exists() else []
    keep_rows = [l for l in lines_tsv[1:] if "\tkeep\t" in l]

    top_results = "\n".join(keep_rows[-10:]) if keep_rows else "(none)"
    date = datetime.now().strftime("%Y-%m-%d")

    content = f"""# Research Conclusion — {date}

## Stop Condition Triggered
{reason}

## Project
{PROJECT_NAME}

## Metric
{METRIC_NAME} | direction: {METRIC_DIRECTION} | goal: {METRIC_GOAL}

## Result
Best {METRIC_NAME}: {best_metric:.6f}
Gap to goal: {(best_metric - METRIC_GOAL) if METRIC_DIRECTION == "lower" else (METRIC_GOAL - best_metric):+.4f}
Goal {"ACHIEVED" if beats_goal(best_metric) else "NOT YET REACHED"}

## Experiments Run
Total: {experiment_n} experiments | Kept: {keeps} | Discarded: {discards}

## Top Results (kept experiments)
```
commit  {METRIC_NAME}  params_M  status  description
{top_results}
```

---

## PROVEN (replicated across experiments)
[Fill in after reviewing skill files in wiki/skills/]

## HYPOTHESIS (one experiment, not yet replicated)
[Fill in]

## SPECULATION (not yet tested)
[Fill in]

## What We Now Know
[2-3 sentences: the key architectural insight from this run]

## Next Research Question
[The single most important open question]
"""
    CONCLUSION.write_text(content)
    log(f"\n{'='*60}")
    log(f"CONCLUSION written to: {CONCLUSION}")
    log(f"{'='*60}\n")
    print(content)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — PLANNER
# Reads: AGENT.md + wiki skills (FTS) + recent results + web
# Outputs: hypothesis + rationale + tags
# ═══════════════════════════════════════════════════════════════════════════════

def agent_planner(current_code: str, best_metric: float, experiment_n: int) -> dict:
    constitution = AGENT_MD.read_text() if AGENT_MD.exists() else ""
    recent = mem.compress_context(load_recent_results(20))

    search_q = f"{METRIC_NAME} {best_metric:.3f} architecture experiment"
    memory_ctx = mem.prefetch(search_q)

    goal_desc = (f"Goal: {METRIC_NAME} {'<' if METRIC_DIRECTION == 'lower' else '>'} {METRIC_GOAL}")

    prompt = f"""You are the PLANNER agent for autonomous ML architecture research.
Your job: propose ONE specific, falsifiable hypothesis to try next.

## Domain
{DOMAIN_CONTEXT}

## Constitution (research rules and accumulated knowledge)
{constitution[:3000]}

## Current State
- Experiment #{experiment_n}
- Best {METRIC_NAME} so far: {best_metric:.6f}
- {goal_desc}

## Recent Experiment History
{recent}

## Memory Context (session FTS + skill manifest cache)
{memory_ctx}

## Current Model Code (first 80 lines for context)
```python
{current_code[:3000]}
```

---

TASK: Propose the single most promising change NOT already tried.
Think from first principles: what does the best model in this domain do that this one doesn't?

Respond in this EXACT JSON format (no markdown, just JSON):
{{
  "hypothesis": "one sentence — what you're changing and why it should help",
  "rationale": "2-3 sentences explaining the mechanism and expected effect",
  "tags": ["tag1", "tag2", "tag3"],
  "web_search_query": "optional: paper or technique to search for, or empty string"
}}
"""
    response = _llm.generate_with_search(prompt) if USE_WEB_SEARCH else _llm.generate_content(prompt)
    text = response.text.strip()

    mem.sync_turn("planner", f"experiment #{experiment_n}, best={best_metric:.4f}", text)

    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    log("[planner] JSON parse failed, using fallback")
    return {
        "hypothesis": text[:200],
        "rationale": "",
        "tags": ["unknown"],
        "web_search_query": ""
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — CODER
# Reads: hypothesis + rationale + current code + constitution
# Outputs: full new model file
# ═══════════════════════════════════════════════════════════════════════════════

def agent_coder(hypothesis: str, rationale: str, current_code: str,
                web_context: str = "") -> tuple[str, str]:
    constitution = AGENT_MD.read_text() if AGENT_MD.exists() else ""
    web_section = f"\n## Web Research Context\n{web_context}\n" if web_context else ""

    prompt = f"""You are the CODER agent for ML architecture research.
The Planner has proposed a hypothesis. Implement it in the model file.

## Domain
{DOMAIN_CONTEXT}

## Hypothesis to implement
{hypothesis}

## Rationale
{rationale}
{web_section}

## Hard Constraints (from constitution — never violate)
{constitution[:1500]}

## Current Code (full — the file you will edit)
```python
{current_code}
```

Implement ONE change. Keep everything else exactly as-is.
Surgical: minimum lines changed, maximum clarity.

Respond in this EXACT format (nothing else):

DESCRIPTION: [max 10 words for the experiment log]

```python
[full updated file with your change applied]
```
"""
    response = model_plain.generate_content(prompt)
    text = response.text

    description = ""
    desc_match = re.search(r'^DESCRIPTION:\s*(.+)$', text, re.MULTILINE)
    if desc_match:
        description = desc_match.group(1).strip()

    code = ""
    code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    return code, description


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR — calls the configured eval script (immutable, never modified)
# Parses metric from stdout: looks for "{METRIC_KEY}=X.XXXX"
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluate() -> dict:
    cmd = [sys.executable, str(EVAL_SCRIPT)] + [str(a) for a in EVAL_ARGS]
    log(f"[eval] Running: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HARNESS_DIR),
                                timeout=7200)  # 2hr hard ceiling — kills hung training
    except subprocess.TimeoutExpired:
        log(f"[eval] TIMEOUT — training hung past 2hr, killing")
        return {METRIC_NAME: worst_possible(), "params_M": 0.0, "status": "crash"}
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr

    print(stdout[-3000:], flush=True)

    if result.returncode != 0:
        log(f"[eval] CRASH ({elapsed:.0f}s): {result.stderr[-300:]}")
        return {METRIC_NAME: worst_possible(), "params_M": 0.0, "status": "crash"}

    # Parse metric from stdout: look for "eval=X.XXXX" or "{METRIC_KEY}=X.XXXX"
    metric_val = None
    train_val = None
    for line in reversed(stdout.splitlines()):
        if f"{METRIC_KEY}=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith(f"{METRIC_KEY}="):
                    try:
                        metric_val = float(p.split("=")[1])
                    except ValueError:
                        pass
                if p.startswith("train="):
                    try:
                        train_val = float(p.split("=")[1])
                    except ValueError:
                        pass
            if metric_val is not None:
                break

    # fallback: try reading last_result.json if the eval script wrote one
    if metric_val is None:
        result_json = EVAL_SCRIPT.parent / "last_result.json"
        if result_json.exists():
            try:
                d = json.loads(result_json.read_text())
                metric_val = d.get("val_bpb") or d.get(METRIC_NAME)
                train_val = d.get("train_bpb")
            except Exception:
                pass

    if metric_val is None:
        log(f"[eval] WARNING: could not parse {METRIC_KEY}= from output")
        metric_val = worst_possible()

    log(f"[eval] {METRIC_NAME}={metric_val:.6f}  goal={METRIC_GOAL}  ({elapsed:.0f}s)")
    return {
        METRIC_NAME: metric_val,
        "train_bpb": train_val,
        "params_M": 0.0,
        "status": "ok",
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — ANALYST
# Reads: hypothesis + result + best so far
# Outputs: keep/discard + writes skill file to wiki
# ═══════════════════════════════════════════════════════════════════════════════

def agent_analyst(hypothesis: str, rationale: str, description: str,
                  result: dict, best_metric: float, tags: list[str]) -> dict:
    metric_val = result.get(METRIC_NAME, worst_possible())
    train_val  = result.get("train_bpb")
    delta      = metric_val - best_metric
    improved   = is_improvement(metric_val, best_metric) and abs(metric_val) < 9.0

    goal_desc = f"goal: {'<' if METRIC_DIRECTION == 'lower' else '>'} {METRIC_GOAL}"

    prompt = f"""You are the ANALYST agent for ML architecture research.
An experiment just finished. Your jobs:
  1. Interpret the result mechanistically
  2. Decide keep or discard
  3. Write a skill file for long-term memory (future Planner agents read this)

## Experiment
Hypothesis: {hypothesis}
Rationale: {rationale}
Description: {description}

## Result
{METRIC_NAME}: {metric_val:.6f}
{"train: " + str(round(train_val, 6)) if train_val else ""}
delta vs best: {delta:+.6f}
previous best: {best_metric:.6f}
{goal_desc}
improved: {improved}

## Domain Context
{DOMAIN_CONTEXT}

---

Respond in this EXACT JSON format:
{{
  "status": "keep" or "discard" or "crash",
  "reason": "1-2 sentence mechanistic explanation of why this result happened",
  "what_worked": "what specifically helped (empty if discard)",
  "what_failed": "why it failed (empty if keep)",
  "skill_content": "3-5 sentence Markdown summary for future agents. Include: what changed, the numbers, mechanistic interpretation, recommendation for follow-up.",
  "follow_up_ideas": ["idea1", "idea2"]
}}
"""
    response = model_plain.generate_content(prompt)
    text = response.text.strip()

    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            raise ValueError("no JSON")
    except Exception:
        outcome = "keep" if improved else "discard"
        analysis = {
            "status": outcome,
            "reason": f"{METRIC_NAME}={metric_val:.4f}, delta={delta:+.4f}",
            "what_worked": "", "what_failed": "",
            "skill_content": f"Tried: {description}. {METRIC_NAME}={metric_val:.4f} delta={delta:+.4f}. Outcome: {outcome}.",
            "follow_up_ideas": []
        }

    # build skill content
    skill_content = analysis.get("skill_content", "")
    if analysis.get("what_worked"):
        skill_content += f"\n\n**What worked:** {analysis['what_worked']}"
    if analysis.get("what_failed"):
        skill_content += f"\n\n**What failed:** {analysis['what_failed']}"
    if analysis.get("follow_up_ideas"):
        skill_content += f"\n\n**Follow-up ideas:** {'; '.join(analysis['follow_up_ideas'])}"

    outcome = analysis.get("status", "discard")
    mem.skill_write(
        name=description or hypothesis[:40],
        tags=tags + ([outcome]),
        outcome=outcome,
        bpb_delta=delta,
        content=skill_content
    )
    mem.sync_turn("analyst", f"{hypothesis} → {description}", text)
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — EVOLVER (GEPA-lite)
# Runs every EVOLVE_EVERY experiments. Rewrites AGENT.md based on skill files.
# ═══════════════════════════════════════════════════════════════════════════════

def agent_evolver(experiment_n: int, best_metric: float):
    log(f"[evolver] Running GEPA evolution at experiment #{experiment_n}...")

    skill_files = sorted(mem.SKILLS_DIR.glob("*.md")) if mem.SKILLS_DIR else []
    if not skill_files:
        log("[evolver] No skill files yet, skipping")
        return

    keeps, discards = [], []
    for sf in skill_files[-40:]:
        content = sf.read_text()
        outcome = "keep" if "outcome: keep" in content else "discard"
        body = content.split("---", 2)[-1].strip()[:300]
        entry = f"[{outcome.upper()}] {sf.stem}: {body}"
        (keeps if outcome == "keep" else discards).append(entry)

    current_constitution = AGENT_MD.read_text() if AGENT_MD.exists() else ""

    prompt = f"""You are the EVOLVER agent. After {experiment_n} experiments, rewrite
the AGENT.md constitution to encode everything learned. Future Planner agents read this.

## Domain
{DOMAIN_CONTEXT}

## Current Constitution
{current_constitution}

## What Worked (KEEP experiments)
{chr(10).join(keeps[-15:]) if keeps else "(none yet)"}

## What Failed (DISCARD experiments)
{chr(10).join(discards[-15:]) if discards else "(none yet)"}

## Current Best
{METRIC_NAME}: {best_metric:.6f} | goal: {METRIC_GOAL}

---

Rewrite AGENT.md:
1. Update "What You CAN Do" — promote levers that produced KEEP results
2. Update "What You CANNOT Do" — add patterns that consistently fail
3. Add/update "Empirical Learnings" section (proven / hypothesis / speculation)
4. Keep all hard constraints from the domain context
5. Keep the same structure and formatting

Output ONLY the full new AGENT.md content, no other text.
"""
    try:
        response = model_plain.generate_content(prompt)
        new_constitution = response.text.strip()
        if len(new_constitution) > 500:
            AGENT_MD.write_text(new_constitution)
            log(f"[evolver] AGENT.md evolved ({len(new_constitution)} chars)")
        else:
            log("[evolver] Evolution output too short, keeping existing AGENT.md")
    except Exception as e:
        log(f"[evolver] Evolution failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH (Planner grounding via Gemini native search)
# ═══════════════════════════════════════════════════════════════════════════════

def web_search_context(query: str) -> str:
    if not query or not USE_WEB_SEARCH:
        return ""
    try:
        log(f"[web] Searching: {query}")
        prompt = f"""Search for and summarize: {query}
Focus on: what the technique is, how it works mechanistically,
and whether it's relevant to: {DOMAIN_CONTEXT[:300]}
Keep summary under 300 words."""
        response = model_search.generate_with_search(prompt)
        return response.text[:1000]
    except Exception as e:
        log(f"[web] Search failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log(f"AUTORESEARCH v2 — {PROJECT_NAME}")
    log(f"Provider: {PROVIDER} | Model: {GEMINI_MODEL} | Council: {_COUNCIL_AVAILABLE}")
    log(f"Metric: {METRIC_NAME} | Goal: {METRIC_DIRECTION} {METRIC_GOAL}")
    log(f"Model file: {SOLUTION.relative_to(ROOT)}")
    log(f"Eval script: {EVAL_SCRIPT.relative_to(ROOT)}")
    log(f"Branch: {git('branch --show-current')}")
    log("=" * 60)

    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(f"commit\t{METRIC_NAME}\tparams_M\tstatus\tdescription\n")

    def load_best():
        if not RESULTS_TSV.exists():
            return worst_possible()
        lines = RESULTS_TSV.read_text().strip().splitlines()
        keeps = [l for l in lines[1:] if "\tkeep\t" in l]
        if not keeps:
            return worst_possible()
        vals = [float(l.split("\t")[1]) for l in keeps]
        return min(vals) if METRIC_DIRECTION == "lower" else max(vals)

    experiment_n = max(1, len(RESULTS_TSV.read_text().strip().splitlines()) - 1
                       if RESULTS_TSV.exists() else 1)
    best_metric = load_best()
    consecutive_failures = 0
    mechanism_counts: dict[str, int] = {}    # tag → count of kept experiments

    # ── MAIN EXPERIMENT LOOP ────────────────────────────────────────────────
    while True:
        log(f"\n{'='*60}")
        log(f"EXPERIMENT #{experiment_n}  |  best={best_metric:.6f}  |  goal={METRIC_GOAL}")
        log(f"Consecutive failures: {consecutive_failures}/{CONSECUTIVE_FAIL_LIMIT}")
        log(f"{'='*60}")

        current_code = SOLUTION.read_text()

        # ── AGENT 1: PLANNER (or Council) ─────────────────────────────────
        if _COUNCIL_AVAILABLE:
            log("[1/4] HYPOTHESIS COUNCIL — 5 personas proposing + chairman synthesizing...")
            plan = run_hypothesis_council(
                domain_context=DOMAIN_CONTEXT,
                constitution=AGENT_MD.read_text() if AGENT_MD.exists() else "",
                best_metric=best_metric,
                metric_name=METRIC_NAME,
                metric_direction=METRIC_DIRECTION,
                metric_goal=METRIC_GOAL,
                recent_results=load_recent_results(15),
                current_code=current_code,
                gemini_model=GEMINI_MODEL,
                provider=PROVIDER,
            )
            if not plan or not plan.get("hypothesis"):
                log("[council] All personas failed — falling back to solo planner")
                plan = agent_planner(current_code, best_metric, experiment_n)
        else:
            log("[1/4] PLANNER — proposing hypothesis (Gemini)...")
            plan = agent_planner(current_code, best_metric, experiment_n)

        hypothesis = plan.get("hypothesis", "")
        rationale  = plan.get("rationale", "")
        tags       = plan.get("tags", [])
        web_q      = plan.get("web_search_query", "")
        log(f"[planner] Hypothesis: {hypothesis}")
        log(f"[planner] Tags: {tags}")

        web_ctx = web_search_context(web_q) if web_q else ""

        # ── AGENT 2: CODER ─────────────────────────────────────────────────
        success = False
        for attempt in range(MAX_RETRIES):
            log(f"[2/4] CODER — implementing (attempt {attempt+1}/{MAX_RETRIES})...")
            new_code, description = agent_coder(hypothesis, rationale, current_code, web_ctx)

            if not new_code.strip():
                log("[coder] Empty output, retrying...")
                continue

            SOLUTION.write_text(new_code)
            check = subprocess.run(
                [sys.executable, "-c",
                 f"compile(open(r'{SOLUTION}').read(), 'cmp_current.py', 'exec')"],
                capture_output=True, text=True
            )
            if check.returncode != 0:
                log(f"[coder] Syntax error: {check.stderr[:200]} — retrying")
                subprocess.run(
                    f"git checkout -- {SOLUTION.relative_to(ROOT)}",
                    shell=True, cwd=str(ROOT), capture_output=True
                )
                continue

            success = True
            break

        if not success:
            log("[coder] All attempts failed — skipping experiment")
            experiment_n += 1
            consecutive_failures += 1
            continue

        log(f"[coder] Description: {description}")
        commit_hash = git_commit(f"autoresearch #{experiment_n}: {description or hypothesis[:50]}")
        log(f"[git] Committed: {commit_hash}")

        # ── EVALUATOR ──────────────────────────────────────────────────────
        log("[3/4] EVALUATOR — running eval script...")
        result = run_evaluate()

        metric_val = result.get(METRIC_NAME, worst_possible())
        params_m   = result.get("params_M", 0.0)
        log(f"[eval] {METRIC_NAME}={metric_val:.6f}  gap_to_goal={metric_val - METRIC_GOAL:+.4f}")

        # ── COUNCIL INTERPRETATION (on surprising results) ─────────────────
        if _COUNCIL_AVAILABLE:
            delta = metric_val - best_metric
            # Only interpret if result is a genuine improvement AND meaningful magnitude.
            # Threshold 0.5 avoids firing on the first real result after a crash run
            # where best_metric = 9.999999 and delta is artificially huge.
            interp_threshold = float(CFG.get("council", {}).get("interpretation_threshold", 0.5))
            is_meaningful = (abs(delta) > interp_threshold and
                             metric_val < 9.0 and best_metric < 9.0)
            if is_meaningful:
                log(f"[council] Surprising result (delta={delta:+.4f}) — running interpretation...")
                interp = run_result_interpretation(
                    domain_context=DOMAIN_CONTEXT,
                    constitution=AGENT_MD.read_text() if AGENT_MD.exists() else "",
                    hypothesis=hypothesis,
                    metric_name=METRIC_NAME,
                    metric_val=metric_val,
                    train_val=result.get("train_bpb", metric_val),
                    best_metric=best_metric,
                    metric_direction=METRIC_DIRECTION,
                    metric_goal=METRIC_GOAL,
                    recent_results=load_recent_results(10),
                    current_code=current_code,
                    gemini_model=GEMINI_MODEL,
                    provider=PROVIDER,
                )
                log(f"[council] Interpretation:\n{interp[:500]}")
                mem.sync_turn("council_interp", hypothesis, interp)

        # ── AGENT 3: ANALYST ───────────────────────────────────────────────
        log("[4/4] ANALYST — interpreting result + writing skill...")
        analysis = agent_analyst(hypothesis, rationale, description,
                                 result, best_metric, tags)

        status = analysis.get("status", "discard")
        reason = analysis.get("reason", "")
        log(f"[analyst] Decision: {status.upper()} | {reason}")

        if status == "keep" and is_improvement(metric_val, best_metric):
            best_metric = metric_val
            consecutive_failures = 0
            log(f"[loop] NEW BEST: {best_metric:.6f}  (gap to goal: {best_metric - METRIC_GOAL:+.4f})")

            # track mechanism confirmation
            for tag in tags:
                mechanism_counts[tag] = mechanism_counts.get(tag, 0) + 1
        else:
            git_reset()
            subprocess.run(
                f"git checkout -- {SOLUTION.relative_to(ROOT)}",
                shell=True, cwd=str(ROOT), capture_output=True
            )
            log("[loop] Reverted to last good commit")
            if status != "crash":
                consecutive_failures += 1

        append_result(commit_hash, metric_val, params_m, status,
                      description or hypothesis[:60])

        # ── GEPA EVOLUTION ─────────────────────────────────────────────────
        if experiment_n % EVOLVE_EVERY == 0:
            agent_evolver(experiment_n, best_metric)

        experiment_n += 1

        # ── STOP CONDITIONS ────────────────────────────────────────────────
        keeps_so_far   = len([l for l in RESULTS_TSV.read_text().splitlines() if "\tkeep\t" in l])
        discards_so_far = experiment_n - 1 - keeps_so_far

        if beats_goal(best_metric):
            write_conclusion(
                f"GOAL ACHIEVED: {METRIC_NAME}={best_metric:.6f} {'<' if METRIC_DIRECTION == 'lower' else '>'} {METRIC_GOAL}",
                best_metric, experiment_n - 1, keeps_so_far, discards_so_far
            )
            sys.exit(0)

        if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
            write_conclusion(
                f"CEILING FOUND: {consecutive_failures} consecutive experiments with no improvement.\n"
                f"Best {METRIC_NAME} reached: {best_metric:.6f}",
                best_metric, experiment_n - 1, keeps_so_far, discards_so_far
            )
            sys.exit(0)

        # mechanism confirmation: same tag improved N times
        for tag, count in mechanism_counts.items():
            if count >= MECHANISM_CONFIRM_N:
                write_conclusion(
                    f"MECHANISM CONFIRMED: '{tag}' produced improvement in {count} independent experiments.",
                    best_metric, experiment_n - 1, keeps_so_far, discards_so_far
                )
                sys.exit(0)


if __name__ == "__main__":
    main()
