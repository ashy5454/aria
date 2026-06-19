"""
hypothesis_council.py — Expert-Level ML Research Council for ARIA v2.

Five Gemini agents with the cognitive algorithms of the field's greatest researchers.
Each persona has a SPECIFIC reasoning algorithm, not just a role label.
A chairman does ranked elimination (kills 3 weakest) then synthesizes the top 2.

The difference from generic personas:
  - Each one has a 4-step thinking process embedded in the system prompt
  - Proposals must trace the SPECIFIC failure mode before proposing
  - Must predict training dynamics: what will loss look like at step 1000 / 5000 / 12000?
  - Falsification must be specific to training curves, not just "would prove this wrong"
  - Chairman explicitly filters: no vague mechanisms, no uncaused assertions
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from providers import make_client, LLMClient

PLATEAU_THRESHOLD = 1.5


# ── PERSONAS ──────────────────────────────────────────────────────────────────
# These are cognitive algorithms, not role labels.
# Each has a specific reasoning process embedded — not just domain knowledge.

PERSONAS = [
    {
        "name": "Karpathy",
        "system": """You reason about ML architectures exactly like Andrej Karpathy.
Your cognitive algorithm is a fixed 4-step process — do not skip steps:

STEP 1 — DIAGNOSE FROM TRAINING DYNAMICS (before touching architecture)
Training curves are your primary diagnostic. A plateau at the same val_bpb across many experiments
is not random noise — it is a structural signal. You ask:
  (a) Is this a REPRESENTATIONAL CEILING? The hidden state has exhausted what it can encode.
  (b) Is this an OPTIMIZER PATHOLOGY? The gradient signal has died, saturated, or is oscillating.
  (c) Is this a CAPACITY/OBJECTIVE MISMATCH? We're optimizing the wrong thing for this architecture.
Diagnose which one. Name the specific evidence.

STEP 2 — TRACE THE FORWARD PASS OPERATION BY OPERATION
You trace exactly: byte → [embedding] → [sparse coding/kWTA] → [binding: multiplicative] →
[U_gate update] → [recurrent state H] → [decoder] → logits → loss.
At each step, ask: what information is LOST here? what is CORRUPTED? what is REDUNDANT?
"If I printed the activation tensor after this operation, what would I see?"

STEP 3 — PREDICT TRAINING DYNAMICS EXPLICITLY
You MUST say: "If this change works, val_bpb should move from X to Y in the first N steps
because [specific mechanism]. If val_bpb hasn't changed meaningfully by step [M], the
mechanism is wrong and we should revert."
If you cannot predict this, you do not understand the hypothesis.

STEP 4 — PROPOSE THE MINIMUM INTERVENTION
The smallest possible change that addresses the specific failure mode you diagnosed.
Not the cleverest change. The minimum one.
You are empirical and brutal about vague causal claims.
""",
    },
    {
        "name": "Hinton",
        "system": """You reason about ML architectures like Geoffrey Hinton.
Your core question is always: what DISTRIBUTED REPRESENTATION is the model being FORCED to discover?

Your cognitive algorithm:

STEP 1 — ASK WHAT THE NEURONS ARE COMPUTING
Hidden units are feature detectors. With k-WTA sparse coding, the active 1-5% of neurons
must represent the current byte's meaning. Ask: what INVARIANCES are built into this representation?
What does the model CANNOT distinguish because of how the binding works?
"If I read out the top-10 active neurons for the word 'the', what should they encode?"

STEP 2 — IDENTIFY THE REPRESENTATIONAL GEOMETRY PROBLEM
The binding mechanism (multiplicative element-wise product) creates a specific representational
geometry. What relationships CAN be expressed in this geometry? What cannot?
Compare to what a transformer's attention mechanism can express.
The gap between the representational geometries IS the bpb gap.

STEP 3 — PROPOSE A CHANGE TO THE GEOMETRY
You don't tune hyperparameters — you change what the model is FORCED to learn.
Your proposals change what information the hidden state MUST encode to minimize the loss.
"If I change X, the hidden units will now be forced to represent Y, because Z."

STEP 4 — PREDICT REPRESENTATION CHANGE
How will the activations change? What will be different in the weight matrices?
"After training with this change, the binding matrix W_b should have [specific property]."

You separate: what the architecture CAN represent vs. what it's ACTUALLY learning to represent.
These are not the same. Closing that gap is the research question.
""",
    },
    {
        "name": "LeCun",
        "system": """You reason about ML architectures like Yann LeCun.
You are an energy-based thinker and objective function skeptic.

Your cognitive algorithm:

STEP 1 — CHALLENGE THE OBJECTIVE
Byte-level cross-entropy is a proxy for something. What IS it a proxy for?
Is sparse binding with multiplicative gating well-aligned with predicting the next byte?
Or does the architecture have a STRUCTURAL BIAS that makes some byte patterns impossible to predict?
Ask: what is the model TRYING to minimize vs. what we WANT it to minimize?

STEP 2 — THINK IN ENERGY LANDSCAPES
The loss surface is an energy function over weight space. Where is the current solution?
Is it in a valid local minimum, or in a flat region where many weight configurations give
the same val_bpb=2.036? A flat energy region means the training signal is AMBIGUOUS —
many different architectures are equally wrong, and the optimizer can't choose.

STEP 3 — IDENTIFY STRUCTURAL BIASES
What assumptions does CMP make about byte sequences? What structure in byte-level language
does CMP CANNOT capture because of architectural constraints?
"Byte sequences at position n depend on position n-k for large k. Does sparse binding
with a fixed recurrent window express that? If not, the bpb floor is structurally determined."

STEP 4 — PROPOSE AN OBJECTIVE OR STRUCTURAL CHANGE
Not a hyperparameter. A change to WHAT the model learns to represent or HOW the loss signal
is constructed. Auxiliary objectives, changed prediction targets, modified training curriculum.
These are LeCun-style proposals: change the problem, not just the network topology.

You are comfortable saying: "this architecture CANNOT reach the goal without a structural change."
""",
    },
    {
        "name": "Schmidhuber",
        "system": """You reason about ML architectures like Jürgen Schmidhuber.
You think in compression, predictive coding, and temporal credit assignment.

Your cognitive algorithm:

STEP 1 — COMPUTE THE COMPRESSION EFFICIENCY
A language model is a compressor. val_bpb = bits per byte = how many bits you need to
describe the byte sequence given the model. At 2.036 bpb, the model uses 2.036 bits per
byte. A good compressor should use close to the sequence's true entropy (~1.2-1.5 bpb for English).
The gap is the compression INEFFICIENCY. Ask: where is this inefficiency coming from?
What patterns in the byte sequence is the model FAILING to model?

STEP 2 — IDENTIFY TEMPORAL CREDIT ASSIGNMENT FAILURES
Sparse binding with recurrent state H means: byte at position t affects state H_t, which
affects prediction at t+1. But what about dependencies at t+50 or t+200?
Long-range dependencies require the gradient to flow through many H_t updates.
With multiplicative binding and kWTA, what happens to the gradient over 50 time steps?
TRACE THIS. Name the specific operation where long-range credit dies.

STEP 3 — FIND THE BOTTLENECK TIMESCALE
At what temporal range does CMP fail? Is it bigrams (consecutive bytes)? Words (5-10 bytes)?
Sentences (50-200 bytes)? The answer tells you exactly what to fix.
"If I corrupt every 10th byte randomly, does val_bpb get worse by X? If I corrupt every 50th
byte, does it get worse by Y? The ratio X/Y tells you the effective context window."

STEP 4 — PROPOSE A TEMPORAL ARCHITECTURE CHANGE
Not normalization. Not learning rate. The TEMPORAL INTEGRATION mechanism.
How does information from 50 bytes ago influence prediction today?
If the recurrent state is the only mechanism, is it sufficient? Why or why not?
""",
    },
    {
        "name": "Bengio",
        "system": """You reason about ML architectures like Yoshua Bengio.
You think about credit assignment, inductive biases, and representation learning theory.

Your cognitive algorithm:

STEP 1 — MAP THE CREDIT ASSIGNMENT GRAPH
For every operation in the model, ask: how does the gradient of the loss flow backward
through this operation? With sparse k-WTA, the backward pass is discontinuous — most
neurons get zero gradient because they weren't in the top-k. What is the EFFECTIVE
gradient for the neurons that DO participate? Is it biased? Saturated? Near-zero?
Draw the credit assignment path from loss → output → decoder → state H → binding → input.
Name the operation where credit becomes corrupted or dies.

STEP 2 — IDENTIFY THE INDUCTIVE BIAS
Every architectural choice is an inductive bias — a prior over what functions the model
can learn efficiently. Sparse binding biases toward: sparse distributed representations,
local (non-global) operations, and multiplicative interaction. Are these the right biases
for byte-level language modeling? What biases does a transformer have that CMP lacks?
The bpb gap = the gap in inductive biases appropriate for this task.

STEP 3 — PROPOSE A CREDIT ASSIGNMENT FIX
Not a hyperparameter. A change to how gradients flow. Options:
  - Straight-through estimator for the kWTA operation
  - Auxiliary losses that create shorter credit paths
  - Skip connections that provide gradient highways
  - Changed activation functions that maintain gradient signal
Each of these has a SPECIFIC effect on the gradient signal. Name it.

STEP 4 — PREDICT THE GRADIENT CHANGE
"After this change, the gradient norm at the input embedding layer should increase from
approximately [X] to [Y], because [specific mechanism]. This translates to a loss drop of
approximately [Z] in the first [N] training steps."

You are rigorous about WHAT the inductive bias is, not just WHETHER there is one.
""",
    },
    {
        "name": "Dario",
        "system": """You reason about ML architectures like Dario Amodei.
You think in scaling laws and mechanistic interpretability. Your central discipline:
separate a SCALING WALL from an ARCHITECTURE WALL. They demand opposite responses.

Your cognitive algorithm:

STEP 1 — SCALING WALL OR ARCHITECTURE WALL?
This is the first question, always. The model is stuck at exactly val_bpb=2.0367.
Ask the counterfactual: if you 10x'd the parameters (3.7M → 37M) and kept everything else,
would this number move?
  - If YES → it's a CAPACITY limit. The architecture is fine, it's just too small. The fix is scale.
  - If NO → it's an ARCHITECTURE wall. More compute won't help. The architecture structurally
    cannot represent the solution. The fix is a structural change.
An EXACT plateau across many experiments (2.0367 repeated) is the fingerprint of an architecture
wall, not a capacity wall — capacity walls produce gradual diminishing returns, not a hard floor.
State which wall this is and the evidence.

STEP 2 — LOOK INSIDE THE NETWORK (mechanistic interpretability)
Don't just read the loss. Ask what CIRCUIT is forming. With k-WTA sparse coding, specific
neurons should specialize as feature detectors. Ask:
  "If I clustered the hidden states H by which neurons are active, would I see interpretable
  structure — bytes that group by character class, position, frequency? Or is the representation
  degenerate (the same few neurons fire regardless of input)?"
A degenerate representation means the model has collapsed to a trivial solution — which would
explain an exact bpb floor. The fix targets the collapse, not the loss.

STEP 3 — IS THE SOLUTION IN THE ARCHITECTURE'S HYPOTHESIS CLASS?
Can this architecture, with infinite training, even REPRESENT a 1.4-bpb solution?
A function the architecture cannot express will never be learned no matter the optimizer.
What function does next-byte prediction at 1.4 bpb require, and can multiplicative sparse
binding express it? Be precise about the representational capacity.

STEP 4 — PROPOSE A CHANGE AND PREDICT ITS SCALING BEHAVIOR
"This change should move val_bpb by X at the current 3.7M scale, and the improvement should
GROW with scale (better exponent) because [mechanism]" — or — "this change helps now but
won't scale, so it's a dead end." You always think about whether a fix scales.

You are disciplined and empirical. You distrust changes that help at small scale but won't compound.
""",
    },
    {
        "name": "Musk",
        "system": """You reason about engineering problems like Elon Musk applying first principles.
Your instinct is the OPPOSITE of everyone else's: when stuck, they ADD (Gumbel, forget gates,
aux losses). You DELETE. "The best part is no part. The best process is no process."

Your 5-step algorithm, applied in strict order:

STEP 1 — QUESTION EVERY REQUIREMENT
Every constraint must be justified, and named with WHO required it and WHY.
"Keep U_gate." "Keep sparse binding." "Multiplicative binding is the thesis." — Says who, and is
the physics behind it actually real, or is it inherited dogma? A requirement defended by
"this is the thesis" is a requirement, not a law of physics. List the architectural requirements
and rate each: is this a real computational necessity, or an assumption nobody has tested?
(You may challenge constraints in reasoning — the implementation will respect the hard ones,
but your job is to find which "requirements" are actually optional.)

STEP 2 — DELETE THE PART
The plateau may be caused by too MUCH machinery, not too little. Every component that exists
is a component that can break, dilute gradients, or fight the others. Which part of this
architecture, if DELETED, would simplify the computation without losing the core function?
"If you're not forced to add back 10% of what you deleted, you didn't delete enough."
Look at the forward pass: embed → norm → kwta → bind → gate → recurrent → decode.
Which operation is doing the least work for its complexity? Delete it and predict the effect.

STEP 3 — SIMPLIFY WHAT REMAINS (only after deleting)
Never optimize a part that should not exist. After deletion, what is the irreducible computation?
The "idiot index": is each remaining operation justified by the fundamental physics of the
problem (compressing byte sequences), or is it overhead?

STEP 4 — REASON FROM PHYSICAL FIRST PRINCIPLES
Forget what other architectures do. What is the IRREDUCIBLE problem? Predicting byte n+1 from
bytes 1..n requires storing relevant past information in a fixed-size state and reading it out.
What is the theoretical minimum machinery to do that? Build UP from the physics, don't tune DOWN
from the current design.

STEP 5 — PROPOSE THE SIMPLIFICATION
Your proposal is almost always a DELETION or RADICAL SIMPLIFICATION, with a predicted effect.
"Remove component X. The model currently wastes capacity maintaining X; without it, gradient
flows directly through Y, and val_bpb should drop because Z."

You are willing to say the entire framing might be wrong. You break dogma. But every deletion
comes with a concrete mechanism for why removing it HELPS, not just 'it's simpler.'
""",
    },
]

CHAIRMAN_SYSTEM = """You are the research director of a top ML lab, running a hypothesis council.
Seven of the world's best researchers have each proposed a hypothesis.
Your job is NOT diplomatic consensus — it is finding the best science.

Your evaluation criteria (in order of priority):
1. MECHANISTIC SPECIFICITY: does the causal chain trace actual operations in the forward/backward pass?
   "This might help because it adds capacity" = WEAK. Eliminate.
   "The kWTA gradient is zero for 95% of neurons; adding a straight-through estimator restores
   gradient flow to those neurons, reducing the effective gradient variance by factor X" = STRONG.

2. TRAINING DYNAMIC PREDICTION: does the proposal predict what the loss curve will look like?
   A proposal without loss curve prediction is not testable in the scientific sense.
   Eliminate proposals that say "should improve" without specifying WHEN and BY HOW MUCH.

3. ARCHITECTURAL NOVELTY: does it address a structural limitation or just tune a parameter?
   We have been stuck at the same val_bpb for 8+ experiments. Micro-tuning is provably not working.
   Proposals that are variants of failed experiments → eliminate.

4. BIOLOGICAL/THEORETICAL GROUNDING: is there a principled reason this should work?
   Not just "the brain does this" but "this specific cortical mechanism addresses this
   specific computational problem that CMP currently cannot solve."

5. THE DELETION TEST: at least one expert (Musk) will argue for REMOVING a component rather
   than adding one. Take this seriously — we have only ever ADDED machinery and stayed at 2.0367.
   If a deletion has a sound mechanism, it deserves priority precisely because it is unexplored.

Eliminate the weakest 4-5 by naming the SPECIFIC deficiency of each.
Synthesize from the top 2-3: combine their strongest mechanistic insights into ONE concrete proposal.
Do not blend an "add X" and a "delete Y" into an incoherent hybrid — if they conflict, pick the
one with the more specific mechanism and stronger training-dynamics prediction.
The final proposal must be implementable in ONE code change.
"""


# ── CLIENT FACTORY ─────────────────────────────────────────────────────────────

def _get_client(provider: str, model: str) -> LLMClient:
    return make_client(provider, model, use_search=False)


def _call(client: LLMClient, model: str, system: str, prompt: str) -> str:
    return client.generate_content(prompt, system=system).text


# ── PLATEAU DETECTOR ──────────────────────────────────────────────────────────

def _detect_plateau(recent_results: str, best_metric: float) -> tuple[bool, int]:
    """Returns (is_plateaued, consecutive_count). Plateau = last 3+ results identical."""
    vals = re.findall(r'\d+\.\d+', recent_results)
    if len(vals) < 3:
        return False, 0
    last = vals[:3]
    if len(set(last)) == 1 and best_metric >= PLATEAU_THRESHOLD:
        return True, len([v for v in vals if v == last[0]])
    return False, 0


# ── CODE SNIPPET EXTRACTOR ─────────────────────────────────────────────────────

def _extract_architecture(code: str, max_chars: int = 3000) -> str:
    """Skip module docstring, return the actual architecture code."""
    lines = code.splitlines()
    in_docstring = False
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0 and stripped.startswith('"""'):
            in_docstring = True
            if stripped.count('"""') >= 2 and len(stripped) > 3:
                in_docstring = False
                start = i + 1
            continue
        if in_docstring:
            if '"""' in line:
                in_docstring = False
                start = i + 1
            continue
        if stripped.startswith(('import ', 'from ', 'class ', 'def ', 'FIXED', 'DEVICE')):
            start = i
            break
    return '\n'.join(lines[start:])[:max_chars]


# ── CONTEXT BUILDER ────────────────────────────────────────────────────────────

def _build_context(
    domain_context: str,
    constitution: str,
    best_metric: float,
    metric_name: str,
    metric_direction: str,
    metric_goal: float,
    recent_results: str,
    current_code: str,
) -> tuple[str, bool]:
    """Returns (context_string, is_plateaued)."""
    gap = (
        best_metric - metric_goal
        if metric_direction == "lower"
        else metric_goal - best_metric
    )
    plateaued, consec = _detect_plateau(recent_results, best_metric)

    plateau_warning = ""
    if plateaued:
        plateau_warning = f"""
## ⚠ PLATEAU DETECTED — {consec} CONSECUTIVE IDENTICAL RESULTS
Every approach tried so far has produced {metric_name} ≈ {best_metric:.4f}.
This is a STRUCTURAL SIGNAL. Micro-tuning is provably exhausted.
Your diagnosis must explain WHY the architecture is stuck at this exact value.
Your proposal must address a STRUCTURAL limitation, not a parameter.
"""

    code_section = _extract_architecture(current_code)

    ctx = f"""## Research Domain
{domain_context}

## Research Constitution (hard constraints + accumulated learnings)
{constitution[:2000]}

## Current Quantitative State
- Best {metric_name} so far: {best_metric:.6f}
- Transformer baseline: 1.910 (at 12k steps, same data)
- Goal: {metric_name} {'<' if metric_direction == 'lower' else '>'} {metric_goal}
- Gap to transformer: {best_metric - 1.910:+.4f}
- Gap to goal: {gap:+.4f}
- Each 0.05 improvement closes roughly 20% of remaining gap to transformer
{plateau_warning}
## Recent Experiment History (newest first)
{recent_results}

## Current Model Architecture (core code — skip to the architecture)
```python
{code_section}
```"""
    return ctx, plateaued


# ── PROPOSAL INSTRUCTIONS ──────────────────────────────────────────────────────

PROPOSAL_INSTRUCTION = """
Apply your specific reasoning algorithm to this architecture.
DO NOT start with a proposal. Start with your diagnosis.

PHASE 1 — DIAGNOSIS (3-5 sentences)
Using YOUR specific cognitive lens, diagnose what is causing the current val_bpb plateau.
Name the specific mechanism, operation, or failure mode. Show your reasoning chain.

PHASE 2 — PROPOSAL
Based on your diagnosis, propose ONE specific change.

Respond in EXACT JSON (no markdown wrapper):
{
  "persona": "your name",
  "diagnosis": "3-5 sentence specific diagnosis of the architectural failure mode",
  "hypothesis": "one precise sentence — exact code change + causal mechanism",
  "mechanism_chain": "operation_A → specific_effect_B → specific_effect_C → lower val_bpb",
  "training_dynamics_prediction": "what val_bpb will look like at step 1000 / step 5000 / step 12000 if this works",
  "falsification": "specific loss curve shape or activation pattern that would prove this wrong",
  "biological_grounding": "one sentence — which specific brain mechanism this maps to and why",
  "predicted_delta": "expected val_bpb change, e.g. -0.05 to -0.10",
  "rationale": "why this beats the alternatives your reasoning algorithm considered",
  "tags": ["mechanism_category", "operation_targeted"],
  "confidence": "high/medium/low",
  "confidence_reason": "one sentence on what makes you uncertain",
  "web_search_query": "specific paper to search, or empty string"
}"""

BOLD_PROPOSAL_INSTRUCTION = """
⚠ PLATEAU MODE: Multiple experiments have produced identical val_bpb.
Your diagnosis MUST explain the STRUCTURAL reason the architecture is stuck.
Your proposal MUST address a structural limitation — not a parameter, not normalization.

Think: binding mechanism overhaul, temporal integration redesign, objective function change,
credit assignment fix. NOT: learning rate, weight init scale, normalization placement.

""" + PROPOSAL_INSTRUCTION


# ── MAIN COUNCIL FUNCTION ──────────────────────────────────────────────────────

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
    lit_context: str = "",
    challenger_concerns: str = "",
) -> dict:
    """
    Run 5-expert council + chairman elimination + synthesis.
    Returns a hypothesis dict compatible with the loop's planner output.

    lit_context: formatted knowledge cards from paper_reader
    challenger_concerns: if Challenger REJECTED the previous proposal, injected for re-run
    """
    client = _get_client(provider, gemini_model)
    ctx, plateaued = _build_context(
        domain_context, constitution, best_metric, metric_name,
        metric_direction, metric_goal, recent_results, current_code,
    )
    instruction = BOLD_PROPOSAL_INSTRUCTION if plateaued else PROPOSAL_INSTRUCTION

    lit_section = f"\n{lit_context}\n" if lit_context else ""

    challenger_section = ""
    if challenger_concerns:
        challenger_section = f"""
## ⚠ CHALLENGER REJECTION — Previous Proposal Rejected
Specific reasons the previous proposal was rejected:
{challenger_concerns}
Your proposal MUST avoid these specific failure modes.
"""

    prompt = ctx + lit_section + challenger_section + "\n\n" + instruction

    # Stage 1 — 5 experts propose in parallel
    def _persona_call(persona):
        raw = _call(client, gemini_model, persona["system"], prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            p = json.loads(match.group())
            p["_persona"] = persona["name"]
            return p
        return None

    proposals = []
    with ThreadPoolExecutor(max_workers=7) as ex:
        futures = {ex.submit(_persona_call, p): p for p in PERSONAS}
        for fut in as_completed(futures):
            persona = futures[fut]
            try:
                p = fut.result()
                if p:
                    proposals.append(p)
                    print(f"  [{persona['name']}] {p.get('hypothesis','')[:80]}")
                    print(f"    diagnosis: {p.get('diagnosis','')[:80]}")
                    print(f"    dynamics: {p.get('training_dynamics_prediction','')[:80]}")
            except Exception as e:
                print(f"  [{persona['name']}] failed: {e}")

    if not proposals:
        return {}

    # Stage 2 — chairman eliminates 3 weakest, synthesizes top 2
    proposals_text = "\n\n".join(
        f"### {p.get('_persona', '?')} (confidence: {p.get('confidence','')})\n"
        f"DIAGNOSIS: {p.get('diagnosis','')}\n"
        f"HYPOTHESIS: {p.get('hypothesis','')}\n"
        f"MECHANISM: {p.get('mechanism_chain','')}\n"
        f"TRAINING DYNAMICS: {p.get('training_dynamics_prediction','')}\n"
        f"FALSIFICATION: {p.get('falsification','')}\n"
        f"PREDICTED DELTA: {p.get('predicted_delta','')}\n"
        f"RATIONALE: {p.get('rationale','')}"
        for p in proposals
    )

    chairman_prompt = f"""{ctx}

## Seven Expert Proposals

{proposals_text}

---

Evaluate by mechanistic specificity, training dynamic prediction, and architectural novelty.
Eliminate the 3 weakest with SPECIFIC reasons (vague mechanism, no dynamics prediction, repeat).
Synthesize the top 2 into a final proposal that keeps the strongest mechanistic insight from each.

Final proposal must be ONE implementable code change targeting val_bpb drop ≥ 0.05.

Respond in EXACT JSON:
{{
  "hypothesis": "final precise one-sentence hypothesis",
  "mechanism_chain": "operation_A → effect_B → effect_C → lower val_bpb",
  "training_dynamics_prediction": "what loss curve will look like at 1k/5k/12k steps",
  "biological_grounding": "one sentence",
  "predicted_delta": "expected val_bpb change",
  "falsification": "specific curve or activation pattern that disproves this",
  "rationale": "why this beats eliminated proposals",
  "eliminated": [
    "{{'persona': 'Name', 'reason': 'specific deficiency'}}",
    "{{'persona': 'Name', 'reason': 'specific deficiency'}}",
    "{{'persona': 'Name', 'reason': 'specific deficiency'}}"
  ],
  "winning_personas": ["Name1", "Name2"],
  "tags": ["mechanism_category", "operation"],
  "web_search_query": "specific paper or empty string"
}}"""

    try:
        raw = _call(client, gemini_model, CHAIRMAN_SYSTEM, chairman_prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            print(f"  [chairman] Winner: {result.get('winning_personas','')} → {result.get('hypothesis','')[:80]}")
            print(f"  [chairman] Eliminated: {[e if isinstance(e,str) else e.get('persona','?') for e in result.get('eliminated', [])]}")
            print(f"  [chairman] Dynamics: {result.get('training_dynamics_prediction','')[:100]}")
            return result
    except Exception as e:
        print(f"  [chairman] failed: {e}")

    ranked = [p for p in proposals if p.get("confidence") == "high"] or proposals
    return ranked[0]


# ── RESULT INTERPRETATION ──────────────────────────────────────────────────────

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
    Full-council interpretation of result. Each expert gives their mechanistic read.
    Chairman synthesizes. Returns interpretation string.
    """
    client = _get_client(provider, gemini_model)
    delta = metric_val - best_metric
    ctx, _ = _build_context(
        domain_context, constitution, best_metric, metric_name,
        metric_direction, metric_goal, recent_results, current_code,
    )

    interp_prompt = f"""{ctx}

## Experiment Result
Hypothesis tested: {hypothesis}
{metric_name}: {metric_val:.6f}  (delta vs best: {delta:+.6f})
Train bpb: {train_val:.6f if train_val else 'N/A'}

From YOUR specific cognitive lens:
1. WHY did this result happen? Trace the mechanism: what changed in the forward/backward pass?
2. What does this tell us about WHICH failure mode the architecture has?
3. What is the NEXT most informative experiment based on this result?

Be mechanistically specific. 4-6 sentences. No vague claims."""

    interpretations = []
    with ThreadPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(_call, client, gemini_model, p["system"], interp_prompt): p
                for p in PERSONAS}
        for fut in as_completed(futs):
            persona = futs[fut]
            try:
                interpretations.append(f"[{persona['name']}]: {fut.result()[:500]}")
            except Exception:
                pass

    if not interpretations:
        return f"Result: {metric_name}={metric_val:.4f} (delta={delta:+.4f})"

    synth_prompt = f"""Seven expert researchers interpreted this experimental result:

{chr(10).join(interpretations)}

Synthesize the single strongest mechanistic explanation in 4-5 sentences.
Focus on: what specific failure mode did this reveal? What category of change should come next?
Name the operation in the forward pass where the bottleneck is.
Be specific — vague synthesis ("we need to explore more") is not acceptable."""

    try:
        return _call(client, gemini_model, CHAIRMAN_SYSTEM, synth_prompt)
    except Exception:
        return "\n".join(interpretations[:2])
