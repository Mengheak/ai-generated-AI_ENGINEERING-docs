# 01 · Role Tracks: ML Engineer vs AI Engineer vs Data Scientist

> **Goal:** Know exactly what each role does and which files in this tutorial to prioritize.

## The Three Roles at a Glance
| | **Data Scientist** | **ML Engineer** | **AI Engineer** |
|---|---|---|---|
| Core question | "What does the data tell us?" | "How do we train & run models reliably at scale?" | "How do we build products on top of foundation models?" |
| Main output | Insights, experiments, models, reports | Production ML systems & pipelines | LLM apps: RAG, agents, copilots |
| Key skills | Statistics, SQL, EDA, A/B testing, communication | Software engineering, MLOps, DL, distributed training | APIs, prompting, RAG, agents, evals, backend |
| Math depth | High (stats) | Medium–High | Medium |
| Coding depth | Medium | High | High |
| Typical tools | SQL, Pandas, sklearn, notebooks, BI tools | PyTorch, Docker, K8s, MLflow, Airflow, cloud | LLM APIs, vector DBs, LangGraph/MCP, FastAPI |

## Priority Map (★★★ = must master, ★ = know basics)
| Tutorial file | Data Sci | ML Eng | AI Eng |
|---|:-:|:-:|:-:|
| 00 Math | ★★★ | ★★ | ★ |
| 00 Python stack | ★★★ | ★★★ | ★★ |
| 01 EDA & features | ★★★ | ★★ | ★ |
| 02 Supervised + evaluation | ★★★ | ★★★ | ★★ |
| 02 Ensembles & tuning | ★★★ | ★★★ | ★ |
| 02 Unsupervised | ★★ | ★★ | ★ |
| 03 Neural nets & training | ★ | ★★★ | ★★ |
| 03 CV / Transformers | ★ | ★★★ | ★★ |
| 04 LLMs & prompting | ★★ | ★★ | ★★★ |
| 04 RAG | ★ | ★★ | ★★★ |
| 04 Fine-tuning | ★ | ★★★ | ★★ |
| 04 Agents | ★ | ★ | ★★★ |
| 04 RL | ★ | ★★ | ★ |
| 05 Serving | ★ | ★★★ | ★★★ |
| 05 MLOps | ★ | ★★★ | ★★ |

## Fast Tracks
**Data Scientist (≈4 months):** 00 → 01 → 02 (all) → SQL deep-dive → A/B testing & causal inference → 04-01 → storytelling with dashboards.

**ML Engineer (≈6 months):** 00 → 01 → 02 → 03 (all) → 04-03 → 05 (all) → Docker/K8s + one cloud (AWS/GCP).

**AI Engineer (≈4 months):** 00-02, 00-03 → 02-01, 02-02 → 03-04 → 04 (01–04) → 05-01 → evals & observability.

> Backend developers (Java/Spring Boot, Node, etc.) have a head start toward **AI Engineer** and **ML Engineer**: APIs, databases, testing, and deployment transfer directly.

## Extra Skills by Role
- **Data Scientist:** advanced SQL (window functions), experimentation (A/B, power analysis), causal inference, Tableau/Power BI.
- **ML Engineer:** data structures & algorithms, distributed training (DDP), GPU optimization, feature stores, streaming (Kafka).
- **AI Engineer:** prompt/eval design, vector DBs, MCP/tool integrations, cost & latency optimization, AI safety/guardrails.

---
Next → [Portfolio Projects](02-portfolio-projects.md)
