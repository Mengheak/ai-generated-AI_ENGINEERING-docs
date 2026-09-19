# 02 · Portfolio Projects

> **Goal:** 3–4 strong, deployed projects beat 20 notebook tutorials. Each should solve a real problem end-to-end.

## What makes a project "hireable"
- Real or messy data (not just Iris/Titanic).
- Clear problem statement + metric + baseline.
- Clean repo: README, structure, tests, `requirements.txt`, Dockerfile.
- **Deployed** (API or demo UI) with a live link or short video.
- Short write-up: what worked, what didn't, what you'd do next.

## Project Ideas by Level
### Beginner
| Project | Skills |
|---|---|
| House/rent price predictor for your city (scraped listings) | EDA, regression, feature engineering |
| Customer churn classifier | Imbalanced classification, threshold tuning, SHAP |

### Intermediate
| Project | Skills |
|---|---|
| Image classifier for local products/food with transfer learning | CNN fine-tuning, augmentation |
| Sales forecasting dashboard | Time series, TimeSeriesSplit, LightGBM |
| Product recommender for an e-commerce store | Embeddings, collaborative filtering |

### Advanced
| Project | Skills | Best for |
|---|---|---|
| RAG assistant over university/government documents (multilingual, e.g., Khmer + English) | Embeddings, pgvector, reranking, evals | AI Eng |
| Agent that queries a real database and creates reports | Tool use, guardrails, observability | AI Eng |
| End-to-end ML platform: training pipeline → MLflow → FastAPI → Docker → drift monitoring | MLOps | ML Eng |
| Fine-tuned small LLM for a narrow task (classification/extraction) vs prompting baseline | QLoRA, evaluation | ML/AI Eng |
| A/B test analysis + causal impact study on public data | Statistics, experimentation | Data Sci |

## Template README
```markdown
# Project Name
**Problem:** one sentence.  **Result:** metric vs baseline (e.g., F1 0.82 vs 0.61).
**Demo:** link / GIF
## Approach
Data → features → model → evaluation → deployment (architecture diagram)
## Run locally
docker compose up
## Lessons learned
```

---
Next → [Interview Checklist](03-interview-checklist.md)
