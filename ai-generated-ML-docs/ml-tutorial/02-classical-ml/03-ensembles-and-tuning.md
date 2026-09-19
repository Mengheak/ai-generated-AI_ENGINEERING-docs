# 03 · Ensembles & Hyperparameter Tuning

> **Goal:** Squeeze the best performance out of tabular data — the core daily skill of ML engineers and data scientists.

## Ensembles
| Method | Idea | Example |
|---|---|---|
| **Bagging** | Train models on bootstrap samples, average → less variance | Random Forest |
| **Boosting** | Train sequentially, each fixes previous errors → less bias | XGBoost, LightGBM, CatBoost |
| **Stacking** | Meta-model learns to combine base models | `StackingClassifier` |

## Gradient Boosting (XGBoost)
```python
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=100, eval_metric="auc",
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print("best iteration:", model.best_iteration)
```
**Tuning order:** `learning_rate` (low, 0.01–0.1) + early stopping → `max_depth` / `num_leaves` → `subsample`, `colsample_bytree` → regularization (`reg_lambda`, `min_child_weight`).

**CatBoost** handles categorical features natively. **LightGBM** is fastest on big data.

## Hyperparameter Search
| Method | Notes |
|---|---|
| Grid search | Exhaustive, slow; fine for 1–2 params |
| Random search | Better than grid for many params |
| Bayesian (Optuna) | Smart search — industry default |

```python
import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "verbose": -1,
    }
    return cross_val_score(LGBMClassifier(**params), X, y, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

## Explainability (SHAP)
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```
Use it to debug features, find leakage, and explain predictions to stakeholders.

## Exercises
1. Beat your Stage-01 Titanic score with LightGBM + Optuna.
2. Build a `StackingClassifier` of logistic regression + RF + XGBoost; compare with CV.

---
Next → [Unsupervised Learning](04-unsupervised-learning.md)
