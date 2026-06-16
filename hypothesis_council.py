"""
hypothesis_council.py — Multi-persona Gemini council for ML architecture research.

Five Gemini 3.5 Flash agents with different reasoning personas propose hypotheses
independently. A sixth acts as chairman and synthesizes the strongest one.

Diversity of perspective > diversity of model. Each persona reasons from a
fundamentally different frame, producing genuine disagreement and stress-testing.

No OpenRouter needed — runs entirely on GEMINI_API_KEY.

Personas:
  Neuroscientist  — grounds hypotheses in biological neural circuits
  Skeptic         — most conservative, highest-probability-of-working change
  Theorist        — information theory, representational geometry, capacity
  Engineer        — numerical stability, gradient flow, training dynamics
  Maverick        — counterintuitive angles, challenges assumptions

Chairman (6th call): synthesizes, stress-tests, picks strongest hypothesis.
"""

from __future__ import annotations

import json
import os
import re

from providers import make_client, LLMClient

# ── personas ──────────────────────────────────────────────────────────────────

PERSONAS = [
    {
        "name": "Neuroscientist",
        "system": (
            "You are a computational neuroscientist with 15 years studying sparse coding, "
            "predictive coding, and cortical circuits. You propose ML architecture changes "
            "grounded in how biological neural circuits actually compute — what inhibition does, "
            "how kWTA maps to lateral inhibition, how the brain handles binding. "
            "You never propose something a real cortical column couldn't do."
        ),
    },
    {
        "name": "Skeptic",
        "system": (
            "You are a rigorous empirical ML researcher. You distrust theory and trust data. "
            "You propose the most conservative, simplest change most likely to work based on "
            "what has actually worked in similar architectures. You avoid complexity. "
            "If something is already working, you ask why before touching it. "
            "Your hypotheses are boring but they land."
        ),
    },
    {
        "name": "Theorist",
        "system": (
            "You are an information theorist and representational geometer. "
            "You think in terms of entropy, mutual information, capacity, and the geometry "
            "of activation spaces. You propose hypotheses grounded in what the math says "
            "should reduce uncertainty in the prediction. You care about why something works "
            "at the level of information flow, not just empirical results."
        ),
    },
    {
        "name": "Engineer",
        "system": (
            "You are a systems engineer focused on training stability and numerical dynamics. "
            "You think about gradient flow, initialization scale, normalization, "
            "loss landscape curvature, and what causes training to diverge or plateau. "
            "You propose changes that fix the training process, not just the architecture. "
            "Your first question is always: where does the gradient die or explode?"
        ),
    },
    {
        "name": "Maverick",
        "system": (
            "You are a creative contrarian who challenges every assumption. "
            "You propose the most unexpected, counterintuitive change — the one no one "
            "would think to try but has a principled reason to work. "
            "You ask: what if the opposite of the obvious thing is true? "
            "You are wrong more often but when you are right, it is a breakthrough."
        ),
    },
]

CHAIRMAN_SYSTEM = (
    "You are the chairman of a research council. Five expert researchers have each proposed "
    "a hypothesis for improving this ML architecture. Your job: read all five, stress-test "
    "each one, identify the strongest, and synthesize a final hypothesis that incorporates "
    "the best insight. You are rigorous, not diplomatic — pick what is most likely to work, "
    "not the most popular answer. Separate proven from speculation."
)

# ── client factory ────────────────────────────────────────────────────────────

def _get_client(provider: str, model: str) -> LLMClient:
    return make_client(provider, model, use_search=False)


def _call(client: LLMClient, model: str, system: str, prompt: str) -> str:
    return client.generate_content(prompt, system=system).text


# ── council context builder ───────────────────────────────────────────────────

def _build_context(
    domain_context: str,
    constitution: str,
    best_metric: float,
    metric_name: str,
    metric_direction: str,
    metric_goal: float,
    recent_results: str,
    current_code_snippet: str,
) -> str:
    gap = (
        best_metric - metric_goal
        if metric_direction == "lower"
        else metric_goal - best_metric
    )
    return f"""## Research Domain
{domain_context}

## Research Constitution (hard constraints + accumulated learnings)
{constitution[:2000]}

## Current State
- Best {metric_name} so far: {best_metric:.6f}
- Goal: {metric_name} {'<' if metric_direction == 'lower' else '>'} {metric_goal}
- Gap remaining: {gap:+.4f}

## Recent Experiment History
{recent_results}

## Current Model Code (first 60 lines)
```python
{current_code_snippet[:2500]}
```"""


PROPOSAL_INSTRUCTION = """
Based on your unique perspective and the context above, propose ONE specific,
falsifiable architectural hypothesis to try next. It must NOT already appear
in the recent experiment history.

Respond in this EXACT JSON format (no markdown, just JSON):
{{
  "hypothesis": "one sentence — what to change and why it should help",
  "rationale": "2-3 sentences explaining the mechanism from your perspective",
  "tags": ["tag1", "tag2"],
  "confidence": "high/medium/low",
  "web_search_query": "optional search query, or empty string"
}}"""


# ── main council functions ────────────────────────────────────────────────────

def run_hypothesis_council(
    domain_context: str,
    constitution: str,
    best_metric: float,
    metric_name: str,
    metric_direction: str,
    metric_goal: float,
    recent_results: str,
    current_code: str,
    gemini_model: str = "gemini-3.5-flash",
    provider: str = "gemini",
) -> dict:
    """
    Run 5-persona Gemini council + chairman synthesis.
    Returns a hypothesis dict compatible with the loop's planner output.
    """
    client = _get_client(provider, gemini_model)
    ctx = _build_context(
        domain_context, constitution, best_metric, metric_name,
        metric_direction, metric_goal, recent_results, current_code,
    )
    prompt = ctx + "\n\n" + PROPOSAL_INSTRUCTION

    # Stage 1 — all 5 personas propose independently
    proposals = []
    for persona in PERSONAS:
        try:
            raw = _call(client, gemini_model, persona["system"], prompt)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                p = json.loads(match.group())
                p["_persona"] = persona["name"]
                proposals.append(p)
                print(f"  [council/{persona['name']}] {p.get('hypothesis','')[:80]}")
        except Exception as e:
            print(f"  [council/{persona['name']}] failed: {e}")

    if not proposals:
        # fallback — return empty so loop falls back to solo planner
        return {}

    # Stage 2 — chairman synthesizes
    proposals_text = "\n\n".join(
        f"### {p['_persona']}\nHypothesis: {p.get('hypothesis','')}\n"
        f"Rationale: {p.get('rationale','')}\nConfidence: {p.get('confidence','')}"
        for p in proposals
    )

    chairman_prompt = f"""{ctx}

## Five Proposals from the Council

{proposals_text}

---

Your job:
1. Identify which proposal has the strongest mechanistic justification
2. Note any critical flaws in the weaker proposals
3. Synthesize a final hypothesis — it can be one of the five, or a synthesis of two

Respond in this EXACT JSON format (no markdown, just JSON):
{{
  "hypothesis": "final one-sentence hypothesis",
  "rationale": "2-3 sentences — mechanistic justification + why this beats the alternatives",
  "tags": ["tag1", "tag2", "tag3"],
  "winning_persona": "which persona's idea won (or 'synthesis')",
  "web_search_query": "optional search query, or empty string"
}}"""

    try:
        raw = _call(client, gemini_model, CHAIRMAN_SYSTEM, chairman_prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            print(f"  [council/chairman] Winner: {result.get('winning_persona','')} → {result.get('hypothesis','')[:80]}")
            return result
    except Exception as e:
        print(f"  [council/chairman] failed: {e}")

    # fallback: return highest-confidence proposal
    ranked = [p for p in proposals if p.get("confidence") == "high"] or proposals
    return ranked[0]


def run_result_interpretation(
    domain_context: str,
    constitution: str,
    hypothesis: str,
    metric_name: str,
    metric_val: float,
    train_val: float,
    best_metric: float,
    metric_direction: str,
    metric_goal: float,
    recent_results: str,
    current_code: str,
    gemini_model: str = "gemini-3.5-flash",
    provider: str = "gemini",
) -> str:
    """
    Two-persona interpretation of a surprising result.
    Skeptic + Theorist give independent reads, chairman synthesizes.
    """
    client = _get_client(provider, gemini_model)
    delta = metric_val - best_metric
    ctx = _build_context(
        domain_context, constitution, best_metric, metric_name,
        metric_direction, metric_goal, recent_results, current_code,
    )

    interp_prompt = f"""{ctx}

## Result to Interpret
Hypothesis tested: {hypothesis}
{metric_name}: {metric_val:.6f}
delta vs best: {delta:+.6f}
train: {train_val:.6f if train_val else 'N/A'}

In 2-3 sentences: WHY did this result happen mechanistically?
What does this tell us about the architecture?"""

    interpretations = []
    for persona in [PERSONAS[1], PERSONAS[2]]:  # Skeptic + Theorist
        try:
            interp = _call(client, gemini_model, persona["system"], interp_prompt)
            interpretations.append(f"[{persona['name']}]: {interp[:400]}")
        except Exception:
            pass

    if not interpretations:
        return f"Result: {metric_name}={metric_val:.4f} (delta={delta:+.4f})"

    synth_prompt = f"""Two researchers interpreted this experimental result:

{chr(10).join(interpretations)}

Synthesize in 2-3 sentences: what is the most likely mechanistic explanation?"""

    try:
        return _call(client, gemini_model, CHAIRMAN_SYSTEM, synth_prompt)
    except Exception:
        return interpretations[0]
