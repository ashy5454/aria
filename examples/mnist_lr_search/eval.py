"""
eval.py — IMMUTABLE eval script for MNIST example.

Trains the MLP from model.py for FIXED_EPOCHS, evaluates on the MNIST test set,
prints: step=N  train=X.XXXX  eval=X.XXXX

The harness reads "eval=X.XXXX" from stdout. Do NOT modify this file.
Agents only modify model.py.

Usage: python eval.py
"""

import sys
from pathlib import Path

# allow importing model.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP, LEARNING_RATE

FIXED_EPOCHS = 5
BATCH_SIZE   = 256
SEED         = 42


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    data_dir = Path(__file__).parent / ".data"
    train_ds = datasets.MNIST(str(data_dir), train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1000,       shuffle=False, num_workers=0)

    model     = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_steps = 0
    for epoch in range(FIXED_EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_steps += 1

        train_loss_avg = train_loss / len(train_loader)

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        val_acc = correct / len(val_ds)

        print(f"step={total_steps:6d}  train={train_loss_avg:.4f}  eval={val_acc:.4f}", flush=True)

    # final line — harness reads this
    print(f"step={total_steps}  train={train_loss_avg:.4f}  eval={val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
