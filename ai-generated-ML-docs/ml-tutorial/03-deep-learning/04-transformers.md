# 04 · Sequences, Attention & Transformers

> **Goal:** Understand the architecture behind all modern AI (LLMs, ViT, speech, multimodal).

## From RNNs to Transformers
| Model | Idea | Problem |
|---|---|---|
| RNN | Process tokens one-by-one with hidden state | Forgets long context, slow |
| LSTM / GRU | Gates to keep memory | Still sequential, hard to scale |
| **Transformer** | **Attention**: every token looks at every other token in parallel | Cost grows O(n²) with length |

## Text → Numbers
1. **Tokenization**: text → subword IDs (BPE / WordPiece).
2. **Embedding**: ID → dense vector.
3. **Positional encoding**: tells the model token order.

## Self-Attention
```
Q = XW_q,  K = XW_k,  V = XW_v
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```
Each token asks (Q) "who is relevant?", compares with keys (K), and takes a weighted mix of values (V). **Multi-head** = several attentions in parallel.

```python
import torch, torch.nn.functional as F
def attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # causal mask for GPT
    return F.softmax(scores, dim=-1) @ V
```

## Transformer Block
```
x → LayerNorm → Multi-Head Attention → + x (residual)
  → LayerNorm → Feed-Forward (MLP)   → + x (residual)
```

## Three Families
| Family | Example | Good at |
|---|---|---|
| Encoder-only | BERT | Classification, embeddings, NER |
| Decoder-only | GPT, Llama, Claude-style LLMs | Text generation, chat, reasoning |
| Encoder–decoder | T5, Whisper | Translation, summarization, speech-to-text |

## Hugging Face in 5 lines
```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
print(clf("This tutorial is really useful!"))
gen = pipeline("text-generation", model="gpt2")
print(gen("Machine learning is", max_new_tokens=20)[0]["generated_text"])
```

## Exercises
1. Implement a single-head attention layer and verify output shapes.
2. Watch/build Karpathy's "Let's build GPT from scratch" (nanoGPT) — the best transformer exercise there is.
3. Fine-tune `distilbert-base-uncased` for sentiment classification with the HF `Trainer`.

---
Next → [LLMs & Prompting](../04-advanced-ai/01-llms-and-prompting.md)
