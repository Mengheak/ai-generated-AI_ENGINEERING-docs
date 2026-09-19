# 02 · Training Deep Nets Well

> **Goal:** Make training stable, fast, and generalize — the difference between a demo and a working model.

## Optimizers & Learning Rate
| Item | Default choice |
|---|---|
| Optimizer | **AdamW** (lr 1e-3 for small nets, 1e-4–5e-5 for fine-tuning) |
| Schedule | Warmup + cosine decay, or `OneCycleLR` |
| Batch size | As large as memory allows; scale lr with it |

**The learning rate is the #1 hyperparameter.** Too high → loss explodes/NaN. Too low → painfully slow.

## Regularization
| Technique | Effect |
|---|---|
| Weight decay | L2-like penalty |
| Dropout (0.1–0.5) | Randomly drops neurons |
| Data augmentation | More varied data (images: flips, crops; text: back-translation) |
| Early stopping | Stop when val loss stops improving |
| Label smoothing | Less over-confident classifiers |

## Normalization & Initialization
- **BatchNorm** (CNNs), **LayerNorm** (transformers) → stable, faster training.
- Residual connections (`x + f(x)`) let very deep nets train.
- PyTorch default init is fine for most cases.

## Speed
```python
scaler = torch.amp.GradScaler("cuda")
for xb, yb in train_dl:
    with torch.autocast(device_type="cuda", dtype=torch.float16):   # mixed precision
        loss = loss_fn(model(xb), yb)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # gradient clipping
    scaler.step(opt); scaler.update()
```
Also: `num_workers` in DataLoader, gradient accumulation for big effective batches, `torch.compile(model)`.

## Debugging Checklist
1. **Overfit one batch first** — if loss doesn't go ~0, the bug is in your code.
2. Check shapes and label dtypes (`long` for CrossEntropy).
3. Loss is NaN → lower lr, clip gradients, check for bad inputs.
4. Val much worse than train → regularize, augment, more data.
5. Plot loss curves every run (TensorBoard / Weights & Biases).
6. Set seeds for reproducibility: `torch.manual_seed(42)`.

## Exercises
1. Add a `CosineAnnealingLR` scheduler and early stopping to your MNIST MLP.
2. Deliberately use lr = 1.0, observe divergence, then fix it.

---
Next → [Computer Vision](03-computer-vision.md)
