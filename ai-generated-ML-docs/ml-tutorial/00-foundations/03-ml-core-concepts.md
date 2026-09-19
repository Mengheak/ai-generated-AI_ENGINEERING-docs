# 03 · How ML Works (Core Concepts)

> **Goal:** Understand the vocabulary and workflow every ML project follows.

## What is ML?
Learning a function `f(X) → y` from data instead of hand-writing rules.

| Type | Data | Examples |
|---|---|---|
| **Supervised** | X + labels y | Price prediction (regression), spam detection (classification) |
| **Unsupervised** | X only | Customer segmentation, anomaly detection |
| **Self-supervised** | X creates its own labels | LLM pretraining (predict next token) |
| **Reinforcement** | Rewards from environment | Game AI, robotics, RLHF |

## The Standard Workflow
```
Problem → Data → EDA → Features → Split → Train → Evaluate → Tune → Deploy → Monitor
```

## Key Ideas
- **Model**: a function with parameters (weights).
- **Loss**: how wrong the model is (MSE, cross-entropy).
- **Optimizer**: updates weights to reduce loss (gradient descent, Adam).
- **Train / Validation / Test split**: train on one, tune on another, report on the last. **Never touch test until the end.**
- **Overfitting**: great on train, bad on new data (memorizing). **Underfitting**: bad on both.
- **Bias–variance tradeoff**: simple models → high bias; complex models → high variance.
- **Regularization**: penalize complexity (L1, L2, dropout).
- **Data leakage**: information from test/future sneaks into training → fake-good results. The #1 real-world bug.
- **Baseline**: always compare against a dumb model (mean, most frequent class).

## Your first model (end-to-end)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

print("baseline:", accuracy_score(y_te, baseline.predict(X_te)))
print("model   :", accuracy_score(y_te, model.predict(X_te)))
```

## The scikit-learn API (learn once, use everywhere)
```python
model.fit(X_train, y_train)     # learn
model.predict(X_new)            # predict
model.predict_proba(X_new)      # class probabilities
transformer.fit_transform(X)    # scalers, encoders
```

## Exercises
1. Sketch train vs validation loss curves for an overfitting model.
2. Train the Iris model with only 5 training samples — what happens?
3. List 3 ways data leakage could happen in a house-price dataset.

---
Next → [EDA & Cleaning](../01-data/01-eda-and-cleaning.md)
