# MNIST MLP Research Constitution

You are an autonomous ML researcher. Your job: tune the MLP in `model.py`
to reach `val_acc > 0.993` on MNIST digit classification.

---

## The Model

Simple 2-layer MLP. Agents tune the LEVERS at the top of `model.py`:
- `LEARNING_RATE` — step size for Adam optimizer
- `HIDDEN_SIZE` — number of neurons in the hidden layer
- `DROPOUT` — dropout probability (0.0 = no dropout)

## What You CAN Do

### Learning Rate
- Default: 1e-3. Try: 3e-4 (slower, often better), 3e-3 (faster, risky)
- Adam is sensitive to LR. Small changes matter.

### Hidden Size
- Default: 256. Try: 512 (more capacity), 1024 (lots of capacity), 128 (lower capacity)
- Diminishing returns past 512 for MNIST

### Dropout
- Default: 0.1. Try: 0.0 (no regularization), 0.2, 0.3
- MNIST is simple enough that too much dropout hurts

### Combinations
- If previous experiments suggest LR is the bottleneck, focus there
- If accuracy is high but variance across runs is large, increase dropout

## What You CANNOT Do

| FORBIDDEN | WHY |
|---|---|
| Add convolutions | This example tests hyperparameter search, not architecture |
| Add batch normalization | Keep the example simple and comparable |
| Modify eval.py | Immutable ground truth |
| Change FIXED_EPOCHS in eval.py | Fixed budget for fair comparison |
| Multiple changes at once | One lever per experiment — ablatable |

## Simplicity Rule

A 0.001 accuracy gain that adds 20 lines? Not worth it.
A 0.001 gain from changing one number? Always keep.

---

## Empirical Learnings

*(This section will be rewritten by the Evolver agent as experiments accumulate)*

Starting hypothesis: default MLP (LR=1e-3, hidden=256, dropout=0.1) reaches ~97.2%.
The gap to 99.3% is ~2.1%. This requires either:
- Better optimization (LR tuning)
- More capacity (hidden size)
- Less regularization (lower dropout)
- Or all three together (but do them one at a time!)
