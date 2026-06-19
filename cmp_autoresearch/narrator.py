"""
narrator.py — Research Narrative Synthesizer for ARIA v2.

Runs every N experiments. Writes a living research story to results/NARRATIVE.md.
Answers: what have we proven, what have we ruled out, what's the bottleneck, what's next.
Dashboard reads NARRATIVE.md and shows it live.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from providers import make_client

NARRATOR_SYSTEM = (
    "You are a scientific narrator writing the ongoing story of an autonomous ML "
    "architecture research session. You write like a Nature Methods lab notebook: "
    "precise, concise, evidence-based. You separate PROVEN (replicated) from "
    "HYPOTHESIS (single experiment) from RULED OUT (failed). "
    "You name specific mechanisms, not vague directions. You are honest about failures."
)


def run_narrator(
    results_tsv: str,
    session_log_tail: str,
    agent_md: str,
    domain_context: str,
    skill_summaries: list[str],
    best_metric: float,
    metric_goal: float,
    experiment_n: int,
    output_path: Path,
    gemini_model: str = "gemini-3.5-flash",
    provider: str = "gemini",
) -> str:
    """
    Generate the research narrative. Writes NARRATIVE.md and returns the text.
    Fails gracefully — returns empty string if Gemini call fails.
    """
    try:
        client = make_client(provider, gemini_model, use_search=False)
    except Exception as e:
        print(f"  [narrator] client init failed: {e}")
        return ""

    keep_rows   = [l for l in results_tsv.splitlines() if "\tkeep\t"    in l]
    discard_rows= [l for l in results_tsv.splitlines() if "\tdiscard\t" in l]
    skill_block = "\n---\n".join(skill_summaries[-12:]) if skill_summaries else "(none yet)"
    date_str    = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt = f"""Synthesize the research story for this CMP autonomous ML session.

## Domain
{domain_context[:1500]}

## Session State
- Experiment number: {experiment_n}
- Best metric (val_bpb): {best_metric:.4f}  |  Goal: {metric_goal}  |  Gap: {best_metric - metric_goal:+.4f}
- Experiments kept: {len(keep_rows)}  |  Discarded: {len(discard_rows)}

## Full Results (TSV)
{results_tsv}

## AGENT.md (accumulated constitution)
{agent_md[:2000]}

## Experiment Skill Summaries
{skill_block}

## Recent Session Log
{session_log_tail}

---

Write the research narrative in EXACTLY this format:

# CMP Research Narrative
*Updated: {date_str} — Experiment #{experiment_n}*

## The Story So Far
[2-3 sentences: what is being built, the thesis, where we are in the search]

## PROVEN (replicated across multiple experiments — treat as architectural fact)
- [bullet: mechanism + evidence number]

## RULED OUT (tried, failed, mechanistic reason)
- [bullet: what was tried + why it failed mechanistically]

## CURRENT HYPOTHESIS (single experiment, not yet replicated)
- [bullet if any]

## Current Bottleneck
[1 focused paragraph: what mechanism is specifically limiting val_bpb right now]

## Most Promising Next Direction
[1 paragraph: what category of change, why, grounded in the evidence above]

## Key Numbers
| Metric | Value |
|--------|-------|
| Best val_bpb | {best_metric:.4f} |
| Goal | {metric_goal} |
| Gap remaining | {best_metric - metric_goal:+.4f} |
| Experiments run | {experiment_n} |
| Kept | {len(keep_rows)} |
| Discarded | {len(discard_rows)} |

Be specific. Name mechanisms. Cite experiment results.
"""

    try:
        response = client.generate_content(prompt)
        narrative = response.text.strip()
        output_path.write_text(narrative)
        print(f"  [narrator] NARRATIVE.md updated ({len(narrative)} chars)")
        return narrative
    except Exception as e:
        print(f"  [narrator] generation failed: {e}")
        return ""
