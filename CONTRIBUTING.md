# Contributing to Crucible

Thanks for wanting to contribute. Crucible is a research instrument — contributions that make the science better are welcome; contributions that add complexity for its own sake are not.

---

## What we want

- **New examples** (`examples/your_domain/`) — the more domains Crucible provably works on, the stronger the thesis
- **Better stop conditions** — smarter ways to detect ceiling, confirmation, or divergence
- **Memory improvements** — better skill extraction, deduplication, cross-session recall
- **Eval script templates** — domain-specific eval interfaces (audio, vision, RL, etc.)
- **Bug fixes** — especially in the council fallback path and CTS bridge

## What we don't want

- New dependencies beyond `google-generativeai httpx pyyaml`
- Changes to the core loop contract (Planner → Coder → Eval → Analyst)
- Breaking changes to `research.yaml` schema without a migration path
- Abstractions that don't have a working example

---

## How to submit

1. Fork the repo
2. Create a branch: `git checkout -b your-feature`
3. Make your change — one thing at a time, like the harness itself enforces
4. Add or update the example that proves the change works
5. Open a PR with: what changed, why, what it proved

## New example checklist

A new `examples/your_domain/` must include:
- [ ] `model.py` — the file agents edit
- [ ] `eval.py` — immutable, prints `eval=X.XXXX`
- [ ] `research.yaml` — fully configured for the domain
- [ ] `AGENT.md` — research constitution with constraints and levers
- [ ] A result — at least one run showing the harness made progress

## Questions

Open an issue with the `question` label. If you're unsure whether your idea fits, open a discussion before writing code.
