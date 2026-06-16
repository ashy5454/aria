"""
model.py — The file agents edit.

This is a simple 2-layer MLP for MNIST digit classification.
Agents modify the LEVERS at the top to improve val_acc.
ONE change per experiment. Keep the MLP class interface intact.

LEVERS (change these):
  LEARNING_RATE  — how fast parameters update
  HIDDEN_SIZE    — capacity of the hidden layer
  DROPOUT        — regularization strength
  EPOCHS         — training budget (careful: eval.py uses FIXED_STEPS to control budget)
"""

import torch
import torch.nn as nn

# ── LEVERS — agents tune these ────────────────────────────────────────────────
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 256
DROPOUT       = 0.1

# ── Architecture (keep the 2-layer MLP structure) ────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
