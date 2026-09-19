# Machine Learning: Beginner → Advanced

A compact, practical path to becoming an **ML Engineer**, **AI Engineer**, or **Data Scientist**.
Each file is short, focused, and ends with exercises. Follow the order, or jump to your role track.

## Roadmap

| Stage | Folder | Level | Time (est.) |
|---|---|---|---|
| 0 | [Foundations](00-foundations/) | Beginner | 3–4 weeks |
| 1 | [Data](01-data/) | Beginner | 2 weeks |
| 2 | [Classical ML](02-classical-ml/) | Intermediate | 4–5 weeks |
| 3 | [Deep Learning](03-deep-learning/) | Intermediate → Advanced | 5–6 weeks |
| 4 | [Advanced AI (LLMs, RAG, Agents, RL)](04-advanced-ai/) | Advanced | 4–6 weeks |
| 5 | [MLOps & Production](05-mlops/) | Advanced | 3–4 weeks |
| 6 | [Career Tracks & Projects](06-career/) | All | Ongoing |

## Full Index

**00 – Foundations**
1. [Math for ML](00-foundations/01-math-for-ml.md)
2. [Python Scientific Stack](00-foundations/02-python-stack.md)
3. [How ML Works (Core Concepts)](00-foundations/03-ml-core-concepts.md)

**01 – Data**
1. [EDA & Data Cleaning](01-data/01-eda-and-cleaning.md)
2. [Feature Engineering & Pipelines](01-data/02-feature-engineering.md)

**02 – Classical ML**
1. [Supervised Learning](02-classical-ml/01-supervised-learning.md)
2. [Model Evaluation](02-classical-ml/02-model-evaluation.md)
3. [Ensembles & Hyperparameter Tuning](02-classical-ml/03-ensembles-and-tuning.md)
4. [Unsupervised Learning](02-classical-ml/04-unsupervised-learning.md)

**03 – Deep Learning**
1. [Neural Networks with PyTorch](03-deep-learning/01-neural-networks-pytorch.md)
2. [Training Deep Nets Well](03-deep-learning/02-training-techniques.md)
3. [Computer Vision (CNNs)](03-deep-learning/03-computer-vision.md)
4. [Sequences, Attention & Transformers](03-deep-learning/04-transformers.md)

**04 – Advanced AI**
1. [LLMs & Prompting](04-advanced-ai/01-llms-and-prompting.md)
2. [RAG (Retrieval-Augmented Generation)](04-advanced-ai/02-rag.md)
3. [Fine-tuning (LoRA / PEFT)](04-advanced-ai/03-fine-tuning.md)
4. [AI Agents & Tool Use](04-advanced-ai/04-agents.md)
5. [Reinforcement Learning Essentials](04-advanced-ai/05-reinforcement-learning.md)

**05 – MLOps**
1. [Serving Models (FastAPI + Docker)](05-mlops/01-model-serving.md)
2. [Experiment Tracking, CI/CD, Monitoring](05-mlops/02-mlops-lifecycle.md)

**06 – Career**
1. [Role Tracks: ML Eng vs AI Eng vs Data Scientist](06-career/01-role-tracks.md)
2. [Portfolio Projects](06-career/02-portfolio-projects.md)
3. [Interview Checklist](06-career/03-interview-checklist.md)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
pip install torch torchvision            # deep learning
pip install transformers datasets peft sentence-transformers  # LLM stage
```

## How to use this tutorial
- **Read → type the code yourself → do the exercises.** Don't copy-paste.
- Each stage ends with a mini project. Put them on GitHub.
- Role-specific shortcut: see [Role Tracks](06-career/01-role-tracks.md).
