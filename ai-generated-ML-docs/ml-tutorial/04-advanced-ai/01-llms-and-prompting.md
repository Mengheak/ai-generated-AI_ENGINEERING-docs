# 01 · LLMs & Prompting

> **Goal:** Understand how LLMs are built and use them reliably through APIs — the core of the AI Engineer role.

## How an LLM is Made
1. **Pretraining**: decoder-only transformer predicts the next token on trillions of tokens.
2. **Supervised fine-tuning (SFT)**: learns to follow instructions from examples.
3. **Preference tuning (RLHF / DPO / RLAIF)**: aligned to be helpful, honest, harmless.
4. **Inference**: generate token by token; sampling controlled by `temperature`, `top_p`.

## Key Concepts
| Term | Meaning |
|---|---|
| Token | ~¾ of an English word; you pay and are limited per token |
| Context window | Max tokens in (prompt + output) |
| Temperature | 0 = deterministic, higher = creative |
| Hallucination | Confident, wrong output → mitigate with RAG, citations, evals |
| Embeddings | Vectors representing meaning → search, clustering, RAG |
| Structured output | Force JSON/schema for reliable parsing |

## Calling an LLM API (Anthropic example)
```python
# pip install anthropic   |   export ANTHROPIC_API_KEY=...
import anthropic
client = anthropic.Anthropic()

resp = client.messages.create(
    model="claude-sonnet-5",           # check docs.claude.com for current model names
    max_tokens=500,
    system="You are a concise ML tutor.",
    messages=[{"role": "user", "content": "Explain overfitting in 2 sentences."}],
)
print(resp.content[0].text)
```
Other providers (OpenAI, Gemini) and local models (Ollama, vLLM) follow the same pattern: messages in → text out.

## Prompt Engineering Patterns
- **Be specific**: role, task, constraints, output format, audience.
- **Few-shot**: include 2–3 input→output examples.
- **Structure with XML tags / delimiters**: `<document>...</document>`.
- **Ask for reasoning** on hard tasks ("think step by step before answering").
- **Force format**: "Respond only with JSON: {\"label\": ..., \"confidence\": ...}" then validate with Pydantic.
- **Chain prompts**: split big tasks into steps (extract → analyze → write).

```python
from pydantic import BaseModel
import json

class Review(BaseModel):
    sentiment: str
    score: int

prompt = """Classify the review. Respond ONLY with JSON: {"sentiment": "positive|negative", "score": 1-5}
<review>Delivery was late but the product is great.</review>"""
raw = client.messages.create(model="claude-sonnet-5", max_tokens=100,
                             messages=[{"role": "user", "content": prompt}]).content[0].text
review = Review(**json.loads(raw))
```

## Evaluation (don't skip!)
- Build a **test set** of 50–200 real inputs with expected outputs.
- Metrics: exact match, rubric scores, LLM-as-judge (with human spot checks).
- Re-run evals on every prompt/model change — treat prompts like code.

## Exercises
1. Build a CLI that classifies support tickets into 5 categories with structured JSON output.
2. Create a 50-example eval set; compare 2 prompts and report accuracy.

---
Next → [RAG](02-rag.md)
