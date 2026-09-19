# 02 · Model Evaluation

> **Goal:** Measure the right thing. A wrong metric ships a wrong model.

## Classification Metrics
|  | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | (TP+TN)/all | Balanced classes only |
| Precision | TP/(TP+FP) | False positives are costly (spam filter) |
| Recall | TP/(TP+FN) | False negatives are costly (disease, fraud) |
| F1 | 2PR/(P+R) | Balance P & R, imbalanced data |
| ROC-AUC | Ranking quality across thresholds | General comparison |
| PR-AUC | Precision–recall area | **Heavily imbalanced** data |
| Log loss | Probability quality | When calibrated probabilities matter |

## Regression Metrics
| Metric | Note |
|---|---|
| MAE | Average absolute error, robust to outliers, same unit as y |
| RMSE | Penalizes big errors more |
| R² | % variance explained (1 = perfect, 0 = predicting the mean) |
| MAPE | % error; breaks when y ≈ 0 |

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:, 1]
print(confusion_matrix(y_te, y_pred))
print(classification_report(y_te, y_pred))
print("ROC-AUC:", roc_auc_score(y_te, y_prob))
```

## Validation Strategies
| Strategy | When |
|---|---|
| Hold-out (train/val/test) | Large data |
| K-Fold CV | Small/medium data |
| Stratified K-Fold | Classification (keeps class ratio) |
| GroupKFold | Same user/patient appears multiple times |
| TimeSeriesSplit | Time-ordered data — **never shuffle** |

## Threshold Tuning
Default threshold is 0.5 — rarely optimal. Choose it on validation data by business cost.
```python
import numpy as np
from sklearn.metrics import precision_recall_curve
p, r, t = precision_recall_curve(y_val, val_prob)
f1 = 2 * p * r / (p + r + 1e-9)
best_threshold = t[np.argmax(f1[:-1])]
```

## Imbalanced Data
- Use PR-AUC / F1 / recall, not accuracy.
- `class_weight="balanced"`, threshold tuning, or resampling (SMOTE via `imbalanced-learn`).

## Diagnose with Learning Curves
- Train high, val low → **overfitting** → more data, regularize, simpler model.
- Both low → **underfitting** → more features, more complex model.

## Exercises
1. On the credit-card fraud dataset, show why 99.8% accuracy can be useless.
2. Pick a threshold that guarantees recall ≥ 0.9 and report precision.

---
Next → [Ensembles & Tuning](03-ensembles-and-tuning.md)
