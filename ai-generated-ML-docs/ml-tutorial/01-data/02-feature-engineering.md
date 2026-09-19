# 02 · Feature Engineering & Pipelines

> **Goal:** Turn raw columns into signals models can use — safely, with no leakage.

## Common Transformations
| Data type | Technique |
|---|---|
| Numeric | Scaling (`StandardScaler`), log for skewed data, binning |
| Categorical (low cardinality) | One-hot encoding |
| Categorical (high cardinality) | Target encoding, frequency encoding, embeddings |
| Dates | Year, month, weekday, is_weekend, days_since |
| Text | TF-IDF (classic) or embeddings (modern) |
| Ratios / interactions | `price / area`, `clicks / impressions` |
| Aggregates | Per-user mean, count, last value (careful with time!) |

**Scaling needed for:** linear models, SVM, KNN, neural nets. **Not needed for:** tree models.

## Pipelines (the professional way)
Pipelines bundle preprocessing + model, so `fit` only learns from training data → no leakage, one object to deploy.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

df = sns.load_dataset("titanic")
num = ["age", "fare", "sibsp", "parch"]
cat = ["sex", "class", "embarked", "who"]
X, y = df[num + cat], df["survived"]      # select features explicitly (no leaky "alive" column)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

clf = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_tr, y_tr)
print("test accuracy:", clf.score(X_te, y_te))
```

Save and reuse the whole thing:
```python
import joblib
joblib.dump(clf, "model.joblib")
clf = joblib.load("model.joblib")
```

## Feature Selection
- Remove near-constant and highly correlated (>0.95) features.
- Use model importance (`feature_importances_`, permutation importance, SHAP).
- Fewer good features > many noisy ones.

## Exercises
1. Add a `family_size = sibsp + parch + 1` feature to the Titanic pipeline. Does it help?
2. Rebuild the pipeline for a regression dataset (House Prices) with `Ridge`.

---
Next → [Supervised Learning](../02-classical-ml/01-supervised-learning.md)
