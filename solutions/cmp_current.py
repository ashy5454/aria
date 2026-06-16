"""
solutions/your_model.py — Replace this with your actual model file.

ARIA will edit THIS file. One change per experiment.

Requirements:
  - Must be a valid Python file
  - Must be runnable by harness/evaluate.py as a subprocess
  - The eval script is what parses results — this file just needs to train and print

See examples/mnist_lr_search/model.py for a complete working example.
"""

# ── Replace everything below with your model ─────────────────────────────────

import argparse
import torch
import torch.nn as nn

def build_model():
    """Build and return your model here."""
    raise NotImplementedError("Replace this file with your actual model.")

def train_and_eval(steps: int, seed: int) -> float:
    """Train for `steps` steps and return the eval metric."""
    raise NotImplementedError("Replace this file with your actual model.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    result = train_and_eval(args.steps, args.seeds)

    # ARIA reads this line — format must match metric_output_key in research.yaml
    print(f"step={args.steps}  train={result:.4f}  eval={result:.4f}")

if __name__ == "__main__":
    main()
