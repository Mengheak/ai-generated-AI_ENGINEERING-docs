# 02 · Experiment Tracking, CI/CD & Monitoring

> **Goal:** Make ML reproducible, automated, and observable in production.

## The MLOps Loop
```
Data → Train → Track → Register → Deploy → Monitor → (drift detected) → Retrain
```

## 1. Experiment Tracking (MLflow)
```python
import mlflow, mlflow.sklearn
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("titanic")
with mlflow.start_run():
    params = {"C": 0.5, "max_iter": 1000}
    model = build_pipeline(**params).fit(X_tr, y_tr)        # your pipeline factory
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    mlflow.log_params(params)
    mlflow.log_metric("val_auc", auc)
    mlflow.sklearn.log_model(model, "model", registered_model_name="titanic-clf")
```
`mlflow ui` → compare runs. Alternatives: Weights & Biases, Neptune.

## 2. Versioning
| What | Tool |
|---|---|
| Code | Git |
| Data | DVC, lakeFS, or versioned object storage |
| Models | MLflow Model Registry |
| Environment | `requirements.txt` / `uv.lock` + Docker |

## 3. CI/CD (GitHub Actions example)
```yaml
name: ml-ci
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt pytest
      - run: pytest -q                     # unit tests + data checks + API tests
      - run: python train.py --check-min-auc 0.80   # fail if model quality regresses
```

## 4. Monitoring in Production
| Monitor | Why |
|---|---|
| Latency, errors, throughput | Standard service health (Prometheus + Grafana) |
| **Data drift** | Input distribution changed (PSI, KS test) |
| **Prediction drift** | Output distribution changed |
| **Model performance** | Once true labels arrive — the real truth |
| Data quality | Nulls, schema changes (Great Expectations, Pandera) |

Tools: Evidently AI, WhyLabs, Arize. For LLM apps: Langfuse, LangSmith (traces, costs, eval scores).

## 5. Orchestration & Scale
- Pipelines: Airflow, Prefect, Dagster, Kubeflow.
- Feature stores: Feast (share features between training & serving → no skew).
- Kubernetes for scaling; cloud ML platforms for managed everything.

## Production Checklist
- [ ] Reproducible training (seeded, versioned data + code)
- [ ] Same preprocessing in training and serving (one pipeline object)
- [ ] Input validation, tests, health endpoint
- [ ] Model registry + rollback plan
- [ ] Drift + performance monitoring with alerts
- [ ] Retraining trigger (schedule or drift-based)

## Exercises
1. Track 10 experiments in MLflow, register the best, and load it in your FastAPI service by name.
2. Generate an Evidently drift report comparing training data vs a shifted copy.

---
Next → [Role Tracks](../06-career/01-role-tracks.md)
