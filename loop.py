"""
loop.py — CMP Autoresearch Loop (Gemini 3.5 Flash)

The fractalsearch pattern for CMP:
  1. Read AGENT.md + current solution
  2. Ask Gemini: "What one change should I try next?"
  3. Apply Gemini's edit to solutions/cmp_current.py
  4. Run harness/evaluate.py
  5. Read val_bpb from harness/last_result.json
  6. Keep (advance) or discard (git reset)
  7. Log to results/results.tsv
  8. Repeat forever

Usage (on VM):
    export GEMINI_API_KEY=your_key_here
    cd /path/to/cmp-research-package
    git checkout -b autoresearch/cmp-jun16
    python cmp_autoresearch/loop.py

Requirements:
    pip install google-generativeai

IMPORTANT: Run in FOREGROUND (no nohup, no &). Use tmux/screen to keep alive.
    tmux new -s cmp_research
    python cmp_autoresearch/loop.py
    # Ctrl+B D to detach, tmux attach -t cmp_research to re-attach
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import google.generativeai as genai

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
HARNESS_DIR = Path(__file__).parent / "harness"
SOLUTIONS_DIR = Path(__file__).parent / "solutions"
RESULTS_DIR = Path(__file__).parent / "results"
AGENT_MD = Path(__file__).parent / "AGENT.md"
SOLUTION = SOLUTIONS_DIR / "cmp_current.py"
RESULT_FILE = HARNESS_DIR / "last_result.json"
RESULTS_TSV = RESULTS_DIR / "results.tsv"

RESULTS_DIR.mkdir(exist_ok=True)
HARNESS_DIR.mkdir(exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"   # Gemini 3.5 Flash (reasoning)
PC_MODE = "off"                      # "off" = v4-identical baseline, "on" = v6 PC
MAX_RETRIES = 3                      # retries if Gemini produces un-runnable code

# ── setup ─────────────────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Set it with: export GEMINI_API_KEY=your_key_here")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(GEMINI_MODEL)


def git(cmd: str) -> str:
    result = subprocess.run(
        f"git {cmd}", shell=True, capture_output=True, text=True, cwd=str(ROOT)
    )
    return result.stdout.strip()


def git_commit(message: str) -> str:
    subprocess.run("git add solutions/cmp_current.py", shell=True, cwd=str(HARNESS_DIR.parent))
    subprocess.run(
        ["git", "commit", "-m", message], cwd=str(ROOT), capture_output=True
    )
    return git("rev-parse --short HEAD")


def git_reset() -> None:
    subprocess.run("git reset --hard HEAD~1", shell=True, cwd=str(ROOT), capture_output=True)


def read_results_tsv() -> list[dict]:
    if not RESULTS_TSV.exists():
        return []
    lines = RESULTS_TSV.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    return [dict(zip(headers, line.split("\t"))) for line in lines[1:]]


def append_result(commit: str, val_bpb: float, params_m: float, status: str, description: str):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tval_bpb\tparams_M\tstatus\tdescription\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{params_m:.2f}\t{status}\t{description}\n")


def run_evaluate(pc: str = PC_MODE) -> dict:
    result = subprocess.run(
        [sys.executable, str(HARNESS_DIR / "evaluate.py"), "--pc", pc],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        print("[loop] evaluate.py crashed:", result.stderr[-1000:], flush=True)
        return {"val_bpb": 9.999999, "params_M": 0.0, "status": "crash"}
    return json.loads(RESULT_FILE.read_text())


def ask_gemini(history: list[dict], current_code: str, best_bpb: float, recent_results: list[dict]) -> str:
    """
    Ask Gemini what change to make next. Returns the full new Python file content.
    """
    agent_constitution = AGENT_MD.read_text()

    recent_summary = ""
    if recent_results:
        recent_summary = "\n".join(
            f"  [{r['status']}] val_bpb={r['val_bpb']}  {r['description']}"
            for r in recent_results[-8:]
        )

    prompt = f"""
{agent_constitution}

---

## Current State

Best val_bpb so far: {best_bpb:.6f}
(Transformer gate = 1.91, we want to beat it)

Recent experiment history:
{recent_summary if recent_summary else "  (no experiments yet — this is the baseline run)"}

## Current Solution Code

```python
{current_code}
```

---

## Your Task

Propose ONE specific architectural change to lower val_bpb.

Rules:
- ONE change only. Don't combine multiple ideas.
- Respect ALL hard constraints in the constitution (no attention, keep kWTA, etc.)
- Choose an idea NOT already tried in the recent history above
- Think from first principles: what does the brain do here?

Respond in this EXACT format (nothing else):

HYPOTHESIS: [one sentence: what you're changing and why you expect it to help]

```python
[full updated Python file — the entire cmp_current.py with your change applied]
```

DESCRIPTION: [short log entry, max 10 words, for results.tsv]
"""

    response = model.generate_content(prompt)
    return response.text


def extract_code_and_meta(gemini_response: str) -> tuple[str, str, str]:
    """
    Parse Gemini's response into (hypothesis, python_code, description).
    Returns ("", "", "") on parse failure.
    """
    hypothesis = ""
    code = ""
    description = ""

    lines = gemini_response.splitlines()

    # hypothesis
    for i, line in enumerate(lines):
        if line.startswith("HYPOTHESIS:"):
            hypothesis = line.replace("HYPOTHESIS:", "").strip()
            break

    # python code block
    in_code = False
    code_lines = []
    for line in lines:
        if line.strip().startswith("```python") and not in_code:
            in_code = True
            continue
        if line.strip() == "```" and in_code:
            in_code = False
            continue
        if in_code:
            code_lines.append(line)
    code = "\n".join(code_lines)

    # description
    for line in lines:
        if line.startswith("DESCRIPTION:"):
            description = line.replace("DESCRIPTION:", "").strip()
            break

    return hypothesis, code, description


def get_params_m(result_json: dict) -> float:
    return result_json.get("params_M", 0.0)


def main():
    print("=" * 60, flush=True)
    print("CMP AUTORESEARCH LOOP — Gemini 3.5 Flash", flush=True)
    print("=" * 60, flush=True)
    print(f"Root: {ROOT}", flush=True)
    print(f"Model: {GEMINI_MODEL}", flush=True)
    print(f"PC mode: {PC_MODE}", flush=True)
    print(f"Branch: {git('branch --show-current')}", flush=True)
    print("=" * 60, flush=True)

    # initialize results.tsv header if needed
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tval_bpb\tparams_M\tstatus\tdescription\n")

    results = read_results_tsv()
    best_bpb = min((float(r["val_bpb"]) for r in results if r["status"] == "keep"), default=9.999)

    experiment_n = len(results) + 1

    while True:
        print(f"\n{'='*60}", flush=True)
        print(f"[loop] Experiment #{experiment_n}  |  best so far: {best_bpb:.6f}", flush=True)
        print(f"{'='*60}", flush=True)

        current_code = SOLUTION.read_text()
        recent_results = read_results_tsv()

        # ask Gemini what to try
        for attempt in range(MAX_RETRIES):
            print(f"[loop] Asking Gemini ({GEMINI_MODEL})...", flush=True)
            try:
                gemini_response = ask_gemini([], current_code, best_bpb, recent_results)
            except Exception as e:
                print(f"[loop] Gemini API error: {e}", flush=True)
                time.sleep(30)
                continue

            hypothesis, new_code, description = extract_code_and_meta(gemini_response)

            if not new_code.strip():
                print(f"[loop] Gemini response parse failed (attempt {attempt+1})", flush=True)
                print("[loop] Raw response:\n", gemini_response[:500], flush=True)
                continue

            print(f"[loop] Hypothesis: {hypothesis}", flush=True)
            print(f"[loop] Description: {description}", flush=True)

            # apply the change
            SOLUTION.write_text(new_code)

            # quick syntax check before running the full eval
            syntax_check = subprocess.run(
                [sys.executable, "-c", f"import ast; ast.parse(open('{SOLUTION}').read())"],
                capture_output=True, text=True
            )
            if syntax_check.returncode != 0:
                print(f"[loop] Syntax error — reverting (attempt {attempt+1})", flush=True)
                subprocess.run(f"git checkout -- solutions/cmp_current.py",
                               shell=True, cwd=str(ROOT))
                continue

            # syntax ok, proceed
            break
        else:
            print("[loop] All attempts failed — skipping experiment", flush=True)
            experiment_n += 1
            continue

        # commit the change
        commit_msg = f"autoresearch #{experiment_n}: {description or hypothesis[:60]}"
        commit_hash = git_commit(commit_msg)
        print(f"[loop] Committed: {commit_hash}", flush=True)

        # run evaluation
        print(f"[loop] Running evaluation (this takes ~20-30 min on T4)...", flush=True)
        result = run_evaluate(pc=PC_MODE)

        val_bpb = result.get("val_bpb", 9.999999)
        params_m = result.get("params_M", 0.0)  # harness may not report this; that's ok

        if result.get("status") == "crash":
            print(f"[loop] CRASH — discarding", flush=True)
            git_reset()
            append_result(commit_hash, 9.999999, params_m, "crash", description or hypothesis[:60])
            experiment_n += 1
            continue

        improved = val_bpb < best_bpb
        status = "keep" if improved else "discard"

        print(f"[loop] val_bpb={val_bpb:.6f}  best={best_bpb:.6f}  → {status.upper()}", flush=True)

        if improved:
            best_bpb = val_bpb
            print(f"[loop] NEW BEST: {best_bpb:.6f}  (vs gate 1.91 → gap={best_bpb-1.91:+.4f})", flush=True)
        else:
            git_reset()
            # restore the current best solution from git
            print(f"[loop] Reverted to last good commit", flush=True)

        append_result(commit_hash, val_bpb, params_m, status, description or hypothesis[:60])
        experiment_n += 1


if __name__ == "__main__":
    main()
