# 01 · Math for ML

> **Goal:** Know *just enough* math to understand what models do. Intuition + NumPy, not proofs.

## 1. Linear Algebra
Data is a matrix `X` (rows = samples, columns = features). Models are mostly matrix ops.

| Concept | Why it matters |
|---|---|
| Vector, matrix | Data, weights |
| Dot product `a·b` | Similarity, a neuron's output |
| Matrix multiply `XW` | A whole layer in one line |
| Transpose `Xᵀ` | Shaping data for ops |
| Norm `‖x‖` | Distance, regularization |
| Eigenvectors / SVD | PCA, dimensionality reduction |

```python
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])   # 3 samples, 2 features
w = np.array([0.5, -1.0])
print(X @ w)              # predictions for each sample -> [-1.5 -2.5 -3.5]
print(np.linalg.norm(w))  # L2 norm
U, S, Vt = np.linalg.svd(X, full_matrices=False)  # used by PCA
```

## 2. Calculus (Optimization)
Training = **minimize a loss** by following the gradient downhill.

- **Derivative**: slope of a function.
- **Gradient** `∇L`: vector of partial derivatives; points uphill.
- **Chain rule**: how backpropagation computes gradients through layers.
- **Gradient descent**: `w ← w − lr · ∇L(w)`

```python
# Minimize L(w) = (w - 3)^2 with gradient descent
w, lr = 0.0, 0.1
for _ in range(50):
    grad = 2 * (w - 3)
    w -= lr * grad
print(round(w, 4))  # ≈ 3.0
```

## 3. Probability & Statistics
| Concept | Used in |
|---|---|
| Mean, variance, std | Scaling, understanding data |
| Distributions (Normal, Bernoulli, Categorical) | Loss functions, noise assumptions |
| Conditional probability, Bayes' rule | Naive Bayes, reasoning about predictions |
| Expectation | Loss = expected error |
| Maximum Likelihood (MLE) | Why MSE and cross-entropy are the "right" losses |
| Hypothesis tests, p-values, confidence intervals | A/B tests (Data Science) |
| Correlation vs causation | Avoiding wrong conclusions |

**Bayes:** `P(A|B) = P(B|A)·P(A) / P(B)`

**Key link:** MSE loss = MLE under Gaussian noise; cross-entropy = MLE for classification.

## Exercises
1. Implement linear regression with gradient descent in pure NumPy on `y = 2x + 1 + noise`.
2. Compute PCA manually with `np.linalg.svd` and project 2D data to 1D.
3. Simulate 10,000 coin flips and plot the running mean (Law of Large Numbers).

**Resources:** 3Blue1Brown (Linear Algebra, Calculus), *Mathematics for Machine Learning* (free PDF), StatQuest.

---
Next → [Python Stack](02-python-stack.md)
