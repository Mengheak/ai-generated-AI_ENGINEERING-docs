# 01 · Supervised Learning

> **Goal:** Know the main algorithms, when to use each, and their key knobs.

## Regression (predict a number)
| Model | Idea | Key params | Use when |
|---|---|---|---|
| Linear Regression | `y = Xw + b` | – | Baseline, interpretability |
| Ridge / Lasso | Linear + L2 / L1 penalty | `alpha` | Many features; Lasso selects features |
| Decision Tree | If/else splits | `max_depth` | Non-linear, interpretable |
| Random Forest / XGBoost | Many trees | see Ensembles | Best default for tabular data |

## Classification (predict a class)
| Model | Idea | Key params | Use when |
|---|---|---|---|
| Logistic Regression | Linear + sigmoid → probability | `C` | Strong baseline, interpretable |
| KNN | Vote of nearest neighbors | `n_neighbors` | Small data, simple |
| Naive Bayes | Bayes rule + independence | – | Text, very fast |
| SVM | Max-margin boundary, kernels | `C`, `kernel`, `gamma` | Small/medium, high-dimensional |
| Decision Tree | Splits by impurity (Gini/entropy) | `max_depth`, `min_samples_leaf` | Explainability |
| Random Forest | Bagged trees | `n_estimators`, `max_depth` | Robust default |
| Gradient Boosting (XGBoost/LightGBM/CatBoost) | Trees fix previous errors | `learning_rate`, `n_estimators`, `max_depth` | **Winner on tabular data** |

## Core Formulas (know these for interviews)
- Linear regression loss (MSE): `L = (1/n) Σ (yᵢ − ŷᵢ)²`
- Logistic: `p = σ(w·x + b)`, `σ(z) = 1 / (1 + e⁻ᶻ)`
- Log loss: `L = −[y log p + (1−y) log(1−p)]`
- Ridge: `MSE + α‖w‖²` · Lasso: `MSE + α‖w‖₁`

## Compare models quickly
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "knn":    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "svm":    make_pipeline(StandardScaler(), SVC()),
    "rf":     RandomForestClassifier(random_state=0),
    "gbm":    HistGradientBoostingClassifier(random_state=0),
}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    print(f"{name:7s} AUC = {s.mean():.3f} ± {s.std():.3f}")
```

## Decision Guide
```
Tabular data?      → Gradient boosting (start with LightGBM/XGBoost), logreg as baseline
Need to explain?   → Linear / logistic / shallow tree (+ SHAP for complex models)
Images/text/audio? → Deep learning (Stage 03) or pretrained models
Tiny dataset?      → Simple models + strong regularization
```

## Exercises
1. Implement logistic regression from scratch in NumPy (sigmoid + gradient descent) and match sklearn's accuracy.
2. Plot a decision tree's train vs validation accuracy as `max_depth` goes from 1 to 20.

---
Next → [Model Evaluation](02-model-evaluation.md)
