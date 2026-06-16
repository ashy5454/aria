"""
harness/evaluate.py — IMMUTABLE eval harness template.

DO NOT MODIFY. ARIA reads results from this script's stdout.
Replace the body of run() with your actual evaluation logic.

Contract:
  - Runs solutions/your_model.py as a subprocess
  - Must print a line containing "{metric_output_key}=X.XXXX" to stdout
  - Returns non-zero exit code on failure

See examples/mnist_lr_search/eval.py for a complete working example.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SOLUTION     = ROOT / "solutions" / "cmp_current.py"
RESULT_FILE  = Path(__file__).parent / "last_result.json"

FIXED_STEPS  = 5000
FIXED_SEED   = 42


def run(seed: int) -> dict:
    t0  = time.time()
    cmd = [
        sys.executable, str(SOLUTION),
        "--steps", str(FIXED_STEPS),
        "--seeds", str(seed),
        "--progress-every", "500",
    ]
    print(f"[harness] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT.parent))
    elapsed = time.time() - t0
    stdout  = result.stdout + result.stderr

    if result.returncode != 0:
        print("[harness] CRASH — stderr:", result.stderr[-2000:], flush=True)
        out = {"eval": 9.999999, "status": "crash", "elapsed_s": elapsed}
        RESULT_FILE.write_text(json.dumps(out, indent=2))
        return out

    # Parse final eval= from stdout
    eval_val = None
    for line in reversed(stdout.splitlines()):
        if "eval=" in line:
            for part in line.split():
                if part.startswith("eval="):
                    try:
                        eval_val = float(part.split("=")[1])
                    except ValueError:
                        pass
            if eval_val is not None:
                break

    if eval_val is None:
        print("[harness] WARNING: could not parse eval= from output", flush=True)
        eval_val = 9.999999

    out = {"eval": eval_val, "status": "ok", "elapsed_s": elapsed}
    RESULT_FILE.write_text(json.dumps(out, indent=2))
    print(f"[harness] eval={eval_val:.6f}  ({elapsed:.0f}s)", flush=True)
    return out


def main():
    ap   = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=FIXED_SEED)
    args = ap.parse_args()
    out  = run(args.seed)
    print(json.dumps(out, indent=2))
    if out["status"] == "crash":
        sys.exit(1)


if __name__ == "__main__":
    main()
