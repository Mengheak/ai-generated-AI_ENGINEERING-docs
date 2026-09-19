# 01 · Neural Networks with PyTorch

> **Goal:** Understand how a neural net learns and write a training loop from memory.

## Concepts
- **Neuron**: `output = activation(w·x + b)`
- **Layer**: many neurons → `activation(XW + b)`
- **Activation** (adds non-linearity): ReLU (default), GELU (transformers), Sigmoid (binary output), Softmax (multi-class output)
- **Forward pass** → predictions → **loss** → **backpropagation** (chain rule computes gradients) → **optimizer step**
- **Epoch**: one pass over the dataset · **Batch**: subset used per step

| Task | Output layer | Loss |
|---|---|---|
| Regression | 1 unit, no activation | `MSELoss` |
| Binary classification | 1 unit (logit) | `BCEWithLogitsLoss` |
| Multi-class | C units (logits) | `CrossEntropyLoss` |

## PyTorch Essentials
```python
import torch
x = torch.randn(3, 4, requires_grad=True)
y = (x ** 2).sum()
y.backward()            # autograd computes dy/dx
print(x.grad)           # = 2x
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## The Training Loop (memorize this)
```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=20, n_classes=3, n_informative=10, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
to_t = lambda a, dt: torch.tensor(a, dtype=dt)
train_dl = DataLoader(TensorDataset(to_t(X_tr, torch.float32), to_t(y_tr, torch.long)), batch_size=64, shuffle=True)
val_dl   = DataLoader(TensorDataset(to_t(X_va, torch.float32), to_t(y_va, torch.long)), batch_size=256)

class MLP(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(20, 3).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(20):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)   # 1. forward + loss
        opt.zero_grad()                 # 2. clear old grads
        loss.backward()                 # 3. backprop
        opt.step()                      # 4. update weights

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            preds = model(xb.to(device)).argmax(1)
            correct += (preds == yb.to(device)).sum().item()
    print(f"epoch {epoch:2d}  val_acc={correct / len(val_dl.dataset):.3f}")

torch.save(model.state_dict(), "mlp.pt")
```

**Remember:** `model.train()` / `model.eval()` (dropout, batchnorm behave differently) and `torch.no_grad()` for inference.

## Exercises
1. Build the same network in pure NumPy (forward + backward) for a 2-layer net on XOR.
2. Train an MLP on MNIST (`torchvision.datasets.MNIST`) to >97% accuracy.

---
Next → [Training Techniques](02-training-techniques.md)
