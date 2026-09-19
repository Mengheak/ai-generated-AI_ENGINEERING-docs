# 02 · Python Scientific Stack

> **Goal:** Be fluent in NumPy, Pandas, Matplotlib — most daily ML work is data handling.

## NumPy — fast arrays
```python
import numpy as np
a = np.arange(12).reshape(3, 4)
a.shape, a.mean(axis=0), a[:, 1], a[a > 5]   # shape, column means, a column, boolean mask
b = a * 2 + 1           # vectorized (no loops!)
a - a.mean(axis=0)      # broadcasting: center each column
```
**Rule:** if you write a Python `for` loop over rows, there's usually a vectorized way.

## Pandas — tabular data
```python
import pandas as pd
df = pd.read_csv("data.csv")
df.head(); df.info(); df.describe()
df["price"].isna().sum()                          # missing values
df = df[df["price"] > 0]                          # filter
df["price_per_m2"] = df["price"] / df["area"]     # new column
df.groupby("city")["price"].agg(["mean", "count"])
df.merge(other_df, on="id", how="left")
pd.get_dummies(df, columns=["city"])              # one-hot encode
```

## Matplotlib / Seaborn — visualization
```python
import matplotlib.pyplot as plt, seaborn as sns
sns.histplot(df["price"], bins=50)
sns.scatterplot(data=df, x="area", y="price", hue="city")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
```

## Must-know tools
| Tool | Use |
|---|---|
| Jupyter / VS Code notebooks | Exploration |
| `venv` / `uv` / `conda` | Environments |
| Git + GitHub | Versioning, portfolio |
| Type hints + functions/modules | Clean, reusable code (ML/AI Engineers) |
| SQL | Pulling data (essential for Data Scientists) |

## Exercises
1. Load Titanic (`sns.load_dataset("titanic")`), compute survival rate by `sex` and `class`.
2. Plot age distribution split by survival.
3. Rewrite a row-by-row loop as a vectorized NumPy op and time both with `%timeit`.

---
Next → [ML Core Concepts](03-ml-core-concepts.md)
