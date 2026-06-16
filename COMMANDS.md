# CMP Autoresearch — VM Run Commands

## One-time setup (run once on VM)

```bash
# Install Gemini SDK
pip install google-generativeai

# Set your API key (add to ~/.bashrc to persist)
export GEMINI_API_KEY=your_23k_credit_key_here
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.bashrc

# Pull latest
cd /path/to/cmp-research-package
git pull origin codex/cmp-l-byte-sparse-v2-audit

# Create autoresearch branch
git checkout -b autoresearch/cmp-jun16
```

---

## Run the loop (every night)

```bash
# Start tmux session (stays alive after SSH disconnect)
tmux new -s cmp

# Run v2 loop (Hermes pattern: 4 agents + wiki + web search + evolution)
python cmp_autoresearch/loop_v2.py

# Detach (keep it running): Ctrl+B  then  D
# Reattach later:           tmux attach -t cmp
# Kill it:                  Ctrl+C inside tmux, or: tmux kill-session -t cmp
```

---

## Watch live progress

```bash
# From your laptop: SSH tunnel to VM port 8000
ssh -L 8000:localhost:8000 user@your-vm-ip

# On VM (new tmux window: Ctrl+B C)
python cmp_autoresearch/dashboard/serve.py
# Open http://localhost:8000 in laptop browser

# Or: just tail the log
tail -f cmp_autoresearch/results/session.log

# Or: watch the results table
watch -n 60 "tail -20 cmp_autoresearch/results/results.tsv"
```

---

## Claude Code commands (run from repo root)

```bash
# Start Claude Code on the VM (SSH in first)
claude

# Then inside Claude Code:
/goal lower CMP val_bpb below 1.91. Read cmp_autoresearch/AGENT.md and loop_v2.py. Run loop_v2.py and monitor.

# Or use /loop for continuous operation:
/loop check cmp_autoresearch/results/results.tsv for latest bpb, read wiki/skills/ for learnings, propose next experiment
```

---

## Read what the agent learned

```bash
# All skill files (what the agent wrote to its long-term memory)
ls cmp_autoresearch/wiki/skills/

# Best experiments
grep "outcome: keep" cmp_autoresearch/wiki/skills/*.md

# Full experiment log
cat cmp_autoresearch/results/results.tsv

# Session log (what happened this run)
cat cmp_autoresearch/results/session.log
```

---

## After overnight run: review learnings

```bash
# What improved bpb
grep "delta: -" cmp_autoresearch/wiki/skills/*.md | head -20

# What the agent evolved AGENT.md to
cat cmp_autoresearch/AGENT.md

# Git history of all experiments
git log --oneline autoresearch/cmp-jun16
```

---

## Config knobs (edit loop_v2.py top section)

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Which Gemini model |
| `PC_MODE` | `off` | `off`=v4 baseline, `on`=v6 prediction-error |
| `EVOLVE_EVERY` | `20` | Rewrite AGENT.md every N experiments |
| `USE_WEB_SEARCH` | `True` | Planner searches papers (uses Gemini grounding) |
| `SKILLS_SEARCH_K` | `5` | Past skills sent to Planner context |

---

## Which loop to run

| Loop | Use when |
|---|---|
| `loop.py` (v1) | Simple, fast, no memory. Good for a first night. |
| `loop_v2.py` (Hermes) | Full cognitive architecture. Memory grows over time. Use for long campaigns. |
