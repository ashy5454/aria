"""
challenger.py — Hypothesis Challenger for ARIA v2.

Stress-tests a hypothesis AFTER the council proposes it and BEFORE the coder implements it.
Catches: constraint violations, broken causal chains, repeated experiments, tiny deltas.
Returns APPROVED / APPROVED_WITH_CONCERNS / REJECTED.

If REJECTED: loop re-runs council with challenger's concerns injected.
Fails gracefully — returns APPROVED if Gemini call fails.
"""

from __future__ import annotations

import json
import re

from providers import make_client

CHALLENGER_SYSTEM = (
    "You are an adversarial peer reviewer for ML architecture research. "
    "Your sole job: find the single weakest point in a proposed hypothesis "
    "before any compute is spent. You are NOT a gatekeeper — good ideas must pass. "
    "You ARE a filter against: (1) hard constraint violations, "
    "(2) broken or hand-wavy causal chains, "
    "(3) changes too small to measure (predicted delta < 0.01 bpb), "
    "(4) exact repeats of already-tried experiments. "
    "If none of these apply, you approve and optionally strengthen the wording."
)


def challenge_hypothesis(
    hypothesis: str,
    mechanism_chain: str,
    predicted_delta: str,
    rationale: str,
    recent_results: str,
    hard_constraints: str,
    gemini_model: str = "gemini-3.5-flash",
    provider: str = "gemini",
) -> dict:
    """
    Returns dict with keys:
      verdict: "APPROVED" | "APPROVED_WITH_CONCERNS" | "REJECTED"
      concerns: list[str]
      constraint_violation: str | None
      causal_chain_flaw: str | None
      repeat_detected: str | None
      strengthened_hypothesis: str
      rejection_reason: str | None
    """
    try:
        client = make_client(provider, gemini_model, use_search=False)
    except Exception as e:
        print(f"  [challenger] client init failed: {e} — auto-approving")
        return _approved(hypothesis)

    prompt = f"""Review this ML architecture hypothesis before implementation.

## Hard Constraints (violation = immediate REJECTED)
{hard_constraints}

## Proposed Hypothesis
{hypothesis}

## Causal Chain (A → B → C → lower bpb)
{mechanism_chain}

## Predicted delta
{predicted_delta}

## Rationale
{rationale}

## Recent Experiment History (check for repeats)
{recent_results}

---

Check in order:
1. Does this violate any hard constraint? (attention? removing U_gate? changing FIXED_STEPS?)
2. Is the causal chain logically sound? Where does it break?
3. Is this a near-exact repeat of a recent experiment?
4. Is the predicted delta < 0.01 (too small to measure reliably)?

Respond in EXACT JSON (no markdown):
{{
  "verdict": "APPROVED" or "APPROVED_WITH_CONCERNS" or "REJECTED",
  "concerns": [],
  "constraint_violation": null,
  "causal_chain_flaw": null,
  "repeat_detected": null,
  "strengthened_hypothesis": "{hypothesis}",
  "rejection_reason": null
}}"""

    try:
        response = client.generate_content(prompt)
        text = response.text.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return _approved(hypothesis)
        result = json.loads(m.group())
        v = result.get("verdict", "APPROVED")
        print(
            f"  [challenger] {v}"
            + (f" — {result.get('rejection_reason','')}" if v == "REJECTED" else "")
            + (f" — concerns: {result.get('concerns','')}" if result.get("concerns") else "")
        )
        return result
    except Exception as e:
        print(f"  [challenger] failed: {e} — auto-approving")
        return _approved(hypothesis)


def _approved(hypothesis: str) -> dict:
    return {
        "verdict": "APPROVED",
        "concerns": [],
        "constraint_violation": None,
        "causal_chain_flaw": None,
        "repeat_detected": None,
        "strengthened_hypothesis": hypothesis,
        "rejection_reason": None,
    }
