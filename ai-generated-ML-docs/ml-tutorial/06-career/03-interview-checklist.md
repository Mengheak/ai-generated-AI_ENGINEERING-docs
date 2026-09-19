# 03 · Interview Checklist

> **Goal:** Be able to answer each item clearly in 1–2 minutes.

## ML Fundamentals (all roles)
- [ ] Bias–variance tradeoff; how to detect and fix over/underfitting
- [ ] L1 vs L2 regularization (and why L1 gives sparsity)
- [ ] Precision vs recall vs F1; ROC-AUC vs PR-AUC; when to use each
- [ ] Cross-validation types; why time series needs special splits
- [ ] Data leakage examples and prevention
- [ ] How gradient boosting works; bagging vs boosting
- [ ] Handling imbalanced data and missing values
- [ ] Explain logistic regression, decision trees, K-Means from scratch

## Deep Learning (ML/AI Eng)
- [ ] Backpropagation and the chain rule
- [ ] Vanishing/exploding gradients and fixes (ReLU, residuals, normalization, clipping)
- [ ] BatchNorm vs LayerNorm; dropout
- [ ] Adam vs SGD; learning-rate schedules
- [ ] Self-attention formula and why `√d_k`
- [ ] Encoder vs decoder vs encoder–decoder transformers

## LLM / AI Engineering
- [ ] Tokenization, context window, temperature
- [ ] RAG pipeline design and how to evaluate it
- [ ] Prompting vs RAG vs fine-tuning — when to choose each
- [ ] LoRA/QLoRA in one sentence each
- [ ] Agent loop, tool calling, guardrails, prompt injection
- [ ] How to evaluate LLM outputs (test sets, LLM-as-judge)
- [ ] Reducing latency and cost (caching, smaller models, batching, quantization)

## Data Science
- [ ] Hypothesis testing, p-values, confidence intervals, statistical power
- [ ] Designing an A/B test end-to-end; common pitfalls (peeking, novelty effect)
- [ ] SQL: joins, GROUP BY, window functions, CTEs
- [ ] Correlation vs causation; confounders
- [ ] Explaining a model result to a non-technical manager

## ML System Design (senior-leaning)
Practice designing: **spam filter · recommender system · fraud detection · search ranking · RAG customer-support bot**.
Structure every answer:
1. Clarify goal & metrics (business + ML)
2. Data sources & labels
3. Features
4. Model choice (baseline → improved)
5. Training & evaluation (offline)
6. Serving (online/batch, latency)
7. Monitoring & retraining
8. Trade-offs & risks

## Coding
- [ ] Python data structures, LeetCode easy–medium (ML Eng especially)
- [ ] Implement from scratch: linear regression, K-Means, attention, a training loop
- [ ] Pandas manipulation under time pressure

**Resources:** *Designing Machine Learning Systems* (Chip Huyen), *Hands-On ML* (Géron), *AI Engineering* (Chip Huyen), fast.ai, Andrej Karpathy's "Zero to Hero" series, Kaggle.

---
🎓 Back to the [Roadmap](../README.md)
