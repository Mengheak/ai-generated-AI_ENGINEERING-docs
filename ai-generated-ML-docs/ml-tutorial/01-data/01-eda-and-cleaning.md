# 01 · EDA & Data Cleaning

> **Goal:** Understand and fix your data before modeling. Garbage in → garbage out.

## EDA Checklist
```python
import pandas as pd, seaborn as sns
df = pd.read_csv("data.csv")

df.shape; df.dtypes; df.head()
df.isna().mean().sort_values(ascending=False)   # % missing per column
df.duplicated().sum()                           # duplicate rows
df.describe(include="all")                      # ranges, weird values
df["target"].value_counts(normalize=True)       # class balance!
df.nunique()                                    # cardinality of categoricals
sns.pairplot(df.sample(500), hue="target")
```

**Questions to answer:**
1. What does one row represent?
2. What is the target? Is it balanced?
3. Which columns are missing / wrong / leaky (contain the answer or future info)?
4. Any outliers or impossible values (age = 250, price < 0)?
5. Is there a time component? (→ split by time, not randomly)

## Cleaning Recipes
| Problem | Fix |
|---|---|
| Missing numeric | Median impute (+ add `is_missing` flag) |
| Missing categorical | `"Unknown"` category or mode |
| Too many missing (>60%) | Consider dropping the column |
| Duplicates | `df.drop_duplicates()` |
| Outliers | Clip (`df[c].clip(lower, upper)`), log-transform, or investigate |
| Wrong types | `pd.to_datetime`, `astype("category")` |
| Inconsistent text | `.str.lower().str.strip()` |

```python
df = df.drop_duplicates()
df["age"] = df["age"].where(df["age"].between(0, 110))   # impossible → NaN
df["income_missing"] = df["income"].isna().astype(int)
df["date"] = pd.to_datetime(df["date"])
```

> ⚠️ **Compute imputation values (median, mean) on the training set only**, then apply to val/test. Use Pipelines (next file) to do this automatically.

## Exercises
1. Run the full EDA checklist on the Kaggle *House Prices* dataset. Write 5 findings.
2. Find one potentially leaky feature in any dataset and explain why.

---
Next → [Feature Engineering](02-feature-engineering.md)
