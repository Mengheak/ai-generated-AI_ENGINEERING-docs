# 02 · RAG (Retrieval-Augmented Generation)

> **Goal:** Let an LLM answer from *your* documents, with fewer hallucinations. The most common production AI pattern.

## Architecture
```
INDEXING:  docs → chunk → embed → store in vector DB
QUERY:     question → embed → retrieve top-k chunks → (rerank) → LLM(question + chunks) → answer + sources
```

## Minimal RAG (no framework)
```python
# pip install sentence-transformers numpy anthropic
import numpy as np, anthropic
from sentence_transformers import SentenceTransformer

docs = open("handbook.txt", encoding="utf-8").read()
chunks = [docs[i:i + 800] for i in range(0, len(docs), 600)]     # 800 chars, 200 overlap

embedder = SentenceTransformer("all-MiniLM-L6-v2")               # use a multilingual model for Khmer etc.
chunk_vecs = embedder.encode(chunks, normalize_embeddings=True)

def retrieve(query, k=4):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vecs @ q                                       # cosine similarity
    return [chunks[i] for i in np.argsort(-scores)[:k]]

def answer(query):
    context = "\n\n".join(f"<source id={i}>{c}</source>" for i, c in enumerate(retrieve(query)))
    prompt = f"""Answer using ONLY the sources. Cite source ids. If the answer isn't there, say you don't know.
{context}
<question>{query}</question>"""
    client = anthropic.Anthropic()
    return client.messages.create(model="claude-sonnet-5", max_tokens=500,
                                  messages=[{"role": "user", "content": prompt}]).content[0].text

print(answer("What is the refund policy?"))
```

## Production Upgrades
| Component | Options / Tips |
|---|---|
| Vector DB | pgvector (Postgres), Qdrant, Chroma, Weaviate, FAISS |
| Chunking | By headings/paragraphs, 300–1000 tokens, overlap 10–20%, keep metadata |
| Hybrid search | BM25 keyword + vector → better recall on names/IDs |
| Reranking | Cross-encoder reranker on top-20 → keep top-5 |
| Query rewriting | LLM rewrites vague/multi-turn questions before retrieval |
| Frameworks | LangChain, LlamaIndex (useful, but understand the raw version first) |

## Evaluating RAG
- **Retrieval**: recall@k, MRR — did the right chunk come back?
- **Generation**: faithfulness (grounded in sources?), answer relevance.
- Tools: RAGAS, custom LLM-as-judge + a labeled question set.

## Common Failures
Bad chunking · wrong embedding model for the language · no metadata filters · too many irrelevant chunks · no "I don't know" instruction.

## Exercises
1. Build RAG over your university/course PDFs with pgvector and a FastAPI endpoint.
2. Add hybrid search + reranking and measure recall@5 before/after on 30 questions.

---
Next → [Fine-tuning](03-fine-tuning.md)
