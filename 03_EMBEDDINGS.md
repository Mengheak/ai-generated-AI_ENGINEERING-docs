# Lesson 3 — Embeddings

> **Goal of this lesson:** Understand embeddings deeply enough to make good engineering decisions: which model, how many dimensions, which similarity metric, how to chunk, how to evaluate retrieval. You'll build a working semantic search engine in pure Python and measure its quality — the exact foundation that Lessons 4 (pgvector) and 5 (RAG) build on.

**Prerequisites:** Lessons 1–2. Basic high-school vector math (we'll review it).

```bash
uv add numpy sentence-transformers scikit-learn matplotlib voyageai openai
```

---

## Table of Contents

1. [The problem embeddings solve](#1-the-problem-embeddings-solve)
2. [Vectors: a 10-minute refresher](#2-vectors-a-10-minute-refresher)
3. [What an embedding is](#3-what-an-embedding-is)
4. [Dimensions](#4-dimensions)
5. [Similarity metrics: cosine, dot product, Euclidean](#5-similarity-metrics-cosine-dot-product-euclidean)
6. [Generating embeddings (local and API)](#6-generating-embeddings-local-and-api)
7. [Semantic search from scratch](#7-semantic-search-from-scratch)
8. [Chunking: what you actually embed](#8-chunking-what-you-actually-embed)
9. [Evaluating retrieval quality](#9-evaluating-retrieval-quality)
10. [Choosing an embedding model](#10-choosing-an-embedding-model)
11. [Beyond search: classification, clustering, dedup](#11-beyond-search-classification-clustering-dedup)
12. [Visualizing embeddings](#12-visualizing-embeddings)
13. [Pitfalls](#13-pitfalls)
14. [Exercises](#14-exercises)
15. [Checklist](#15-checklist)

---

## 1. The problem embeddings solve

A user of a mart app searches:

> "something to clean my kitchen floor"

Your product table contains:

> "Mop with microfiber head — ideal for tiles"

**Keyword search** (`LIKE '%kitchen floor%'`, or even full-text search) finds nothing: no words overlap. **Semantic search** finds it because the *meanings* are close.

Embeddings make this possible by turning text into points in space where **distance reflects meaning**:

```
"something to clean my kitchen floor"   → [0.12, -0.48, 0.33, ...]  ┐
"Mop with microfiber head"              → [0.10, -0.51, 0.29, ...]  ┘ close together
"Chocolate chip cookies 200g"           → [-0.62, 0.20, -0.11, ...]    far away
```

Embeddings power: semantic search, RAG retrieval, recommendations, deduplication, clustering, classification, anomaly detection, and semantic caching.

---

## 2. Vectors: a 10-minute refresher

A vector is an ordered list of numbers. In 2D, `v = [3, 4]` is an arrow from the origin to the point (3, 4).

### Length (L2 norm)

$$\|v\| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}$$

`[3, 4]` has length 5.

### Dot product

$$a \cdot b = \sum_i a_i b_i = \|a\|\|b\|\cos\theta$$

The dot product is large when vectors point in the same direction, zero when perpendicular, negative when opposite.

### Normalization

Dividing by length gives a **unit vector** (length 1) pointing the same way:

$$\hat{v} = \frac{v}{\|v\|}$$

```python
import numpy as np

a = np.array([3.0, 4.0])
b = np.array([6.0, 8.0])     # same direction, twice as long
c = np.array([-4.0, 3.0])    # perpendicular to a

print(np.linalg.norm(a))                 # 5.0
print(a @ b)                             # 50.0  (dot product)
print(a @ c)                             # 0.0   (perpendicular)
print(a / np.linalg.norm(a))             # [0.6 0.8]
```

That's all the math you need. Embeddings are the same thing with hundreds or thousands of dimensions.

---

## 3. What an embedding is

An **embedding model** is a neural network (usually a transformer, like the LLMs in Lesson 1) trained so that **semantically similar inputs map to nearby vectors**.

```
text ──► tokenizer ──► transformer ──► pooling ──► (optional) normalize ──► vector[d]
```

- **Pooling** combines per-token vectors into one vector (e.g., average them).
- Training typically uses **contrastive learning**: show the model pairs that should be close (a question and its answer, a title and its article) and pairs that should be far, and adjust weights to pull/push them.

### Embedding model vs. generative LLM

| | Embedding model | Generative LLM |
|---|---|---|
| Output | One fixed-size vector | A sequence of tokens |
| Purpose | Compare / retrieve | Generate / reason |
| Cost | Very cheap | Much more expensive |
| Speed | Milliseconds, batchable | Seconds |
| Typical size | ~20M–7B parameters | Much larger |

### Individual dimensions don't have clean meanings

It's tempting to think "dimension 17 = food-ness." In practice, meaning is **distributed** across all dimensions. What matters is **relative position** — the geometry between vectors — not any single number.

### Embeddings are model-specific

Vectors from model A and model B live in **different spaces**. Comparing them is meaningless, even if they have the same dimension. If you switch models, you must **re-embed everything**. Store the model name next to every vector (Lesson 4).

---

## 4. Dimensions

The **dimension** `d` is the vector length, fixed per model: e.g., 384, 768, 1024, 1536, 3072.

### Trade-offs

| More dimensions | Fewer dimensions |
|---|---|
| Can capture finer distinctions | Less expressive |
| More storage: `d × 4 bytes` (float32) per vector | Cheaper storage |
| Slower similarity computation and indexing | Faster search |

Storage math you should do for every project:

```
1,000,000 chunks × 1536 dims × 4 bytes = 6.1 GB (vectors only, before indexes)
1,000,000 chunks ×  384 dims × 4 bytes = 1.5 GB
```

Vector indexes (HNSW, Lesson 4) add significant extra memory on top, and they perform best when they fit in RAM.

### Matryoshka embeddings (shortening vectors)

Some modern models are trained so that the **first k dimensions are themselves a good embedding** ("Matryoshka Representation Learning"). You can truncate a 1536-d vector to 512-d and renormalize with a small quality loss. Several APIs expose this with a `dimensions` / `output_dimension` parameter.

```python
def truncate_and_normalize(vec: np.ndarray, k: int) -> np.ndarray:
    v = vec[:k]
    return v / np.linalg.norm(v)
```

⚠️ Only do this with models that document Matryoshka support. Truncating an ordinary model's vectors destroys quality.

### Quantization

Store vectors as `float16`, `int8`, or even binary to cut memory 2–32×, often with a re-ranking step on full-precision vectors to recover quality. pgvector supports `halfvec` (float16) and binary quantization — an advanced optimization for Lesson 10-scale systems.

---

## 5. Similarity metrics: cosine, dot product, Euclidean

### 5.1 Cosine similarity

Measures the **angle** between vectors, ignoring length.

$$\text{cos\_sim}(a,b) = \frac{a \cdot b}{\|a\| \|b\|} \in [-1, 1]$$

- `1.0` → same direction (very similar)
- `0.0` → unrelated
- `-1.0` → opposite (rare in practice for text embeddings; most text similarities fall in a narrower positive range)

**Cosine distance** = `1 − cosine similarity` (what pgvector's `<=>` returns).

### 5.2 Dot product (inner product)

$$a \cdot b = \sum a_i b_i$$

Affected by both angle **and** length.

### 5.3 Euclidean (L2) distance

$$\|a - b\| = \sqrt{\sum (a_i - b_i)^2}$$

Straight-line distance. Smaller = more similar.

### 5.4 The key insight: normalize, and they all agree

If vectors are **unit-normalized**:

- cosine similarity = dot product
- Euclidean distance² = `2 − 2 × cosine similarity`

So **all three produce the same ranking**. Normalizing once at write time lets you use the fastest operation (dot product) everywhere.

```python
# similarity.py
import numpy as np

def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidean(a, b):
    return float(np.linalg.norm(a - b))

rng = np.random.default_rng(0)
a, b = rng.normal(size=384), rng.normal(size=384)
an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)

print("cosine        :", cosine_similarity(a, b))
print("dot(normed)   :", float(an @ bn))                       # identical to cosine
print("euclid² normed:", euclidean(an, bn) ** 2)
print("2 - 2*cosine  :", 2 - 2 * cosine_similarity(a, b))       # identical
```

### 5.5 Which to use?

**Use whatever the embedding model was trained for** (check its model card). Most text embedding models are designed for cosine similarity and many return normalized vectors already. When in doubt: normalize + cosine.

### 5.6 Vectorized similarity for many documents

Never loop in Python over thousands of vectors. Use matrix math:

```python
def top_k(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int = 5):
    """doc_matrix: shape (n_docs, d), rows already normalized; query_vec normalized."""
    scores = doc_matrix @ query_vec            # (n_docs,) all cosine sims at once
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]   # O(n) partial sort
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
```

This brute-force (**exact**) search over 1 million 384-d vectors takes well under a second on a laptop. Beyond that, or with low-latency requirements, you need **approximate nearest neighbor (ANN)** indexes — Lesson 4.

---

## 6. Generating embeddings (local and API)

### 6.1 Local: sentence-transformers

Free, private, runs on CPU. Great for learning and many production workloads.

```python
# embed_local.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # English, 384 dims
print("dimension:", model.get_sentence_embedding_dimension())

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "What time does the store open?",
]
emb = model.encode(sentences, normalize_embeddings=True)   # numpy array (3, 384)
print(emb.shape)

sims = emb @ emb.T
for i, s in enumerate(sentences):
    print(f"{s!r:40} → " + "  ".join(f"{x:.2f}" for x in sims[i]))
```

Expected pattern: sentences 0 and 1 are highly similar despite sharing almost no words; sentence 2 is far from both.

**Multilingual (important for Khmer/English apps):** English-only models produce poor vectors for Khmer. Use a multilingual model and test it on your own data:

```python
ml = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # 384 dims

pairs = [
    "Where is the nearest pharmacy?",
    "ឱសថស្ថានជិតបំផុតនៅឯណា?",        # same meaning in Khmer
    "The football match starts at 7pm.",
]
e = ml.encode(pairs, normalize_embeddings=True)
print(e @ e.T)   # check whether the English↔Khmer pair scores higher than the unrelated sentence
```

Other strong open multilingual options to evaluate include the `intfloat/multilingual-e5-*` family and `BAAI/bge-m3`. **Always benchmark on your data** — support for lower-resource languages varies widely.

> **E5-style models need prefixes:** documents as `"passage: ..."`, queries as `"query: ..."`. Forgetting this silently reduces quality. Read the model card of every model you use.

### 6.2 API: Voyage AI

Anthropic doesn't offer its own embedding model and recommends Voyage AI. Voyage models distinguish **query** vs **document** inputs:

```python
# embed_voyage.py
import voyageai

vo = voyageai.Client()   # reads VOYAGE_API_KEY

docs = ["Mop with microfiber head — ideal for tiles", "Chocolate chip cookies 200g"]
doc_vecs = vo.embed(docs, model="voyage-3.5", input_type="document").embeddings

query_vec = vo.embed(["something to clean my kitchen floor"], model="voyage-3.5",
                     input_type="query").embeddings[0]
print(len(query_vec))
```

### 6.3 API: OpenAI

```python
from openai import OpenAI
client = OpenAI()

r = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Mop with microfiber head", "Chocolate chip cookies 200g"],
    dimensions=512,            # Matryoshka-style shortening supported by this model family
)
vectors = [d.embedding for d in r.data]
print(len(vectors[0]), r.usage.total_tokens)
```

> Check each provider's docs for current model names, dimensions, and pricing — they change regularly.

### 6.4 A reusable embedder interface

```python
# llm/embeddings.py
from typing import Protocol
import numpy as np

class Embedder(Protocol):
    name: str
    dim: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...

class LocalEmbedder:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 query_prefix="", doc_prefix="", batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._m = SentenceTransformer(model_name)
        self.dim = self._m.get_sentence_embedding_dimension()
        self.qp, self.dp, self.bs = query_prefix, doc_prefix, batch_size

    def embed_documents(self, texts):
        return self._m.encode([self.dp + t for t in texts], batch_size=self.bs,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=len(texts) > 500).astype(np.float32)

    def embed_query(self, text):
        return self._m.encode(self.qp + text, normalize_embeddings=True).astype(np.float32)

class VoyageEmbedder:
    def __init__(self, model="voyage-3.5", dim=1024, batch_size=128):
        import voyageai
        self._c = voyageai.Client()
        self.name, self.dim, self.bs = model, dim, batch_size

    def _normalize(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=-1, keepdims=True)

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), self.bs):
            out.extend(self._c.embed(texts[i:i + self.bs], model=self.name, input_type="document").embeddings)
        return self._normalize(out)

    def embed_query(self, text):
        return self._normalize(self._c.embed([text], model=self.name, input_type="query").embeddings[0])
```

We'll use `Embedder` in Lessons 4 and 5.

---

## 7. Semantic search from scratch

Let's build a real in-memory search engine over a product catalog and compare it with keyword search.

```python
# semantic_search.py
import re
import numpy as np
from llm.embeddings import LocalEmbedder

PRODUCTS = [
    {"id": 1, "name": "Microfiber floor mop", "desc": "Spin mop with bucket, ideal for tiles and wooden floors."},
    {"id": 2, "name": "Dish soap lemon 750ml", "desc": "Cuts grease on plates, pans and cutlery."},
    {"id": 3, "name": "Jasmine rice 5kg", "desc": "Fragrant Cambodian jasmine rice, premium grade."},
    {"id": 4, "name": "Instant noodles chicken", "desc": "Quick meal ready in 3 minutes."},
    {"id": 5, "name": "Mosquito repellent spray", "desc": "Protects skin for up to 8 hours outdoors."},
    {"id": 6, "name": "Paracetamol 500mg", "desc": "Relief for headache and mild fever."},
    {"id": 7, "name": "Toilet cleaner gel", "desc": "Removes stains and kills germs in the bathroom."},
    {"id": 8, "name": "Rechargeable desk fan", "desc": "USB fan, keeps you cool during power cuts."},
    {"id": 9, "name": "Iced coffee sachets", "desc": "Khmer-style sweet coffee with condensed milk flavor."},
    {"id": 10, "name": "Glass cleaner spray", "desc": "Streak-free shine for windows and mirrors."},
]

def doc_text(p):  # what we embed: combine fields that carry meaning
    return f"{p['name']}. {p['desc']}"

embedder = LocalEmbedder()
doc_matrix = embedder.embed_documents([doc_text(p) for p in PRODUCTS])   # (10, 384)

def semantic_search(query: str, k: int = 3):
    q = embedder.embed_query(query)
    scores = doc_matrix @ q
    order = np.argsort(-scores)[:k]
    return [(PRODUCTS[i]["name"], float(scores[i])) for i in order]

def keyword_search(query: str, k: int = 3):
    q_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for p in PRODUCTS:
        words = set(re.findall(r"\w+", doc_text(p).lower()))
        overlap = len(q_words & words)
        if overlap:
            scored.append((p["name"], overlap))
    return sorted(scored, key=lambda x: -x[1])[:k]

queries = [
    "something to clean my kitchen floor",
    "I have a headache",
    "it's too hot and the electricity is off",
    "bugs keep biting me at night",
    "a drink to wake me up",
]
for q in queries:
    print(f"\nQ: {q}")
    print("  keyword :", keyword_search(q))
    print("  semantic:", [(n, round(s, 3)) for n, s in semantic_search(q)])
```

You'll see keyword search returning nothing or wrong results for most queries, while semantic search finds the mop, paracetamol, fan, repellent, and coffee.

### Why semantic search is not enough on its own

Semantic search is weak at **exact matches**: product codes (`SKU-44821`), error codes (`ORA-01017`), names, and rare technical terms. "Paracetamol 500mg" vs "Paracetamol 1000mg" may embed almost identically. Production systems use **hybrid search** (keyword + vector) — you'll implement it in Postgres in Lesson 4.

### Similarity thresholds

Top-k always returns *something*, even for irrelevant queries ("tell me a joke" → still returns 3 products). Add a minimum score:

```python
def semantic_search_threshold(query, k=3, min_score=0.35):
    return [(n, s) for n, s in semantic_search(query, k) if s >= min_score]
```

⚠️ Score scales **differ between models**. A good threshold for one model may be terrible for another. Calibrate it using the evaluation method in section 9.

---

## 8. Chunking: what you actually embed

You can't meaningfully embed a 50-page PDF as one vector — the meaning gets averaged into mush, and models have input limits. You split documents into **chunks**. Chunking is one of the highest-impact decisions in RAG.

### 8.1 Strategies

| Strategy | How | Good for |
|---|---|---|
| Fixed-size | Every N tokens/characters, with overlap | Quick baseline |
| Recursive | Split by paragraphs → sentences → words until under size | Most general text (default choice) |
| Structure-aware | Split by headings, sections, Markdown/HTML structure, code functions | Docs, manuals, code |
| Semantic | Split where embedding similarity between sentences drops | Long unstructured prose |
| Parent–child | Embed small chunks, return the larger parent section | Precise matching + rich context |

### 8.2 A recursive chunker with overlap

```python
# chunking.py
def recursive_chunk(text: str, max_chars: int = 1000, overlap: int = 150,
                    separators=("\n\n", "\n", ". ", " ")) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        chunks, current = [], ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_chars:       # a single part still too big → go deeper
                    chunks.extend(recursive_chunk(part, max_chars, overlap, separators[1:] or (" ",)))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return add_overlap(chunks, overlap)

    # no separators found: hard split
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]

def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append(prev[-overlap:] + " " + cur)
    return out
```

### 8.3 Chunk metadata and contextual headers

A chunk like `"It must be submitted within 7 days."` is meaningless alone. **Prepend context** before embedding:

```python
def contextualize(chunk: str, doc_title: str, section: str) -> str:
    return f"Document: {doc_title}\nSection: {section}\n\n{chunk}"

contextualize("It must be submitted within 7 days.",
              "KhmerMart Return Policy", "Refund requests")
```

This simple trick often improves retrieval noticeably. A more advanced version uses an LLM to write a one-sentence context for each chunk ("contextual retrieval").

### 8.4 Size guidelines

- Start around **300–800 tokens per chunk** with **10–20% overlap**, then tune using evaluation.
- Smaller chunks → more precise matches, less context per hit.
- Larger chunks → more context, but diluted embeddings and more tokens sent to the LLM.
- Never split in the middle of a table row or code block if you can avoid it.

---

## 9. Evaluating retrieval quality

"It seems to work" is not engineering. Build a small labeled dataset and measure.

### 9.1 The dataset

For each query, list the IDs of relevant documents:

```python
EVAL_SET = [
    {"query": "something to clean my kitchen floor", "relevant": {1}},
    {"query": "I have a headache", "relevant": {6}},
    {"query": "it's too hot and the electricity is off", "relevant": {8}},
    {"query": "bugs keep biting me at night", "relevant": {5}},
    {"query": "a drink to wake me up", "relevant": {9}},
    {"query": "make my windows shiny", "relevant": {10}},
    {"query": "cleaning products", "relevant": {1, 2, 7, 10}},
    {"query": "what can I cook quickly for dinner", "relevant": {4, 3}},
]
```

Start with 30–100 realistic queries. Get them from real user logs when you have them.

### 9.2 Metrics

| Metric | Question it answers |
|---|---|
| **Recall@k** | Of all relevant docs, what fraction appear in the top k? (Most important for RAG — if it's not retrieved, the LLM can't use it.) |
| **Precision@k** | Of the top k results, what fraction are relevant? |
| **Hit rate@k** | Did at least one relevant doc appear in top k? |
| **MRR** | How high is the first relevant result? (1/rank, averaged) |
| **nDCG@k** | Rank-aware quality with graded relevance |

```python
# eval_retrieval.py
import numpy as np

def evaluate(search_fn, eval_set, k=3):
    recalls, precisions, hits, rr = [], [], [], []
    for item in eval_set:
        retrieved = search_fn(item["query"], k)          # list of doc ids, best first
        relevant = item["relevant"]
        found = [d for d in retrieved if d in relevant]

        recalls.append(len(found) / len(relevant))
        precisions.append(len(found) / k)
        hits.append(1.0 if found else 0.0)
        rank = next((i + 1 for i, d in enumerate(retrieved) if d in relevant), None)
        rr.append(1 / rank if rank else 0.0)

    return {f"recall@{k}": np.mean(recalls), f"precision@{k}": np.mean(precisions),
            f"hit_rate@{k}": np.mean(hits), "mrr": np.mean(rr)}

def semantic_ids(query, k):
    q = embedder.embed_query(query)
    order = np.argsort(-(doc_matrix @ q))[:k]
    return [PRODUCTS[i]["id"] for i in order]

for k in (1, 3, 5):
    print(k, evaluate(semantic_ids, EVAL_SET, k))
```

Now you can answer questions with data:
- Does the multilingual model beat MiniLM on my data?
- Does adding the description to the embedded text help?
- Does chunk size 500 beat 1000?
- Is Voyage worth paying for vs. a local model?

### 9.3 Calibrating a threshold

Collect scores for relevant pairs and for irrelevant pairs, then pick a threshold that separates them:

```python
rel_scores, irrel_scores = [], []
for item in EVAL_SET:
    q = embedder.embed_query(item["query"])
    scores = doc_matrix @ q
    for i, p in enumerate(PRODUCTS):
        (rel_scores if p["id"] in item["relevant"] else irrel_scores).append(float(scores[i]))

print("relevant   : mean %.3f min %.3f" % (np.mean(rel_scores), np.min(rel_scores)))
print("irrelevant : mean %.3f max %.3f" % (np.mean(irrel_scores), np.max(irrel_scores)))
```

---

## 10. Choosing an embedding model

Decision factors, in rough priority order:

1. **Quality on your data and language** (measured with section 9, not leaderboard vibes). Public benchmarks like MTEB are a starting shortlist only.
2. **Language coverage** — critical for Khmer.
3. **Max input length** — must exceed your chunk size.
4. **Dimension** — storage and speed.
5. **Privacy / deployment** — can data leave your infrastructure?
6. **Cost & throughput** — API price per token vs. GPU/CPU cost for local.
7. **Stability** — API models may be deprecated; local models are frozen files you control.
8. **Domain** — code, legal, medical, and finance-specialized models exist.

A practical path:

```
Prototype:  local multilingual sentence-transformers model (free, fast iteration)
Evaluate:   compare 2–3 local + 1–2 API models on 50+ real queries
Production: choose the cheapest model within ~2–3% of the best recall@k
```

---

## 11. Beyond search: classification, clustering, dedup

### 11.1 Zero-shot classification by label similarity

```python
labels = {
    "delivery": "Problems or questions about delivery time, drivers, or location",
    "payment": "Payment failed, ABA/KHQR, refunds, charges",
    "product_quality": "Damaged, expired, or wrong products",
    "account": "Login, password, profile, account settings",
}
label_names = list(labels)
label_vecs = embedder.embed_documents(list(labels.values()))

def classify(text):
    scores = label_vecs @ embedder.embed_query(text)
    return label_names[int(np.argmax(scores))], float(np.max(scores))

print(classify("The driver called me but couldn't find my house in Sen Sok"))
print(classify("My KHQR payment went through twice"))
```

### 11.2 Trained classifier on embeddings (better with labeled data)

```python
from sklearn.linear_model import LogisticRegression

texts  = ["driver is late", "wrong address used", "card payment failed", "charged twice",
          "milk expired", "box was crushed", "can't log in", "reset my password"]
y      = ["delivery", "delivery", "payment", "payment",
          "product_quality", "product_quality", "account", "account"]

X = embedder.embed_documents(texts)
clf = LogisticRegression(max_iter=1000).fit(X, y)
print(clf.predict(embedder.embed_documents(["the rider went to the wrong street"])))
```

Embeddings + logistic regression is fast, cheap, and often competitive with calling an LLM for classification at scale.

### 11.3 Near-duplicate detection

```python
def find_duplicates(texts, threshold=0.92):
    E = embedder.embed_documents(texts)
    S = E @ E.T
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if S[i, j] >= threshold:
                pairs.append((texts[i], texts[j], round(float(S[i, j]), 3)))
    return pairs
```

(For large datasets, use an ANN index instead of the O(n²) matrix.)

### 11.4 Clustering

```python
from sklearn.cluster import KMeans

feedback = ["app crashes on login", "cannot sign in", "delivery late again",
            "rider was rude", "love the fresh vegetables", "fruit quality is great",
            "login button broken", "order took 3 hours"]
E = embedder.embed_documents(feedback)
km = KMeans(n_clusters=3, n_init="auto", random_state=0).fit(E)
for c in range(3):
    print(c, [f for f, l in zip(feedback, km.labels_) if l == c])
```

A great real-world pattern: cluster thousands of support tickets, then ask an LLM to name each cluster.

---

## 12. Visualizing embeddings

Project high-dimensional vectors to 2D to build intuition (distances are distorted, so use it for exploration only):

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

texts = [doc_text(p) for p in PRODUCTS] + queries
E = embedder.embed_documents(texts)
xy = PCA(n_components=2).fit_transform(E)

plt.figure(figsize=(9, 7))
n = len(PRODUCTS)
plt.scatter(xy[:n, 0], xy[:n, 1], marker="o", label="products")
plt.scatter(xy[n:, 0], xy[n:, 1], marker="x", label="queries")
for (x, y), t in zip(xy, [p["name"] for p in PRODUCTS] + queries):
    plt.annotate(t[:28], (x, y), fontsize=8)
plt.legend(); plt.title("Products and queries (PCA)"); plt.tight_layout()
plt.savefig("embeddings_pca.png")
```

UMAP or t-SNE usually give clearer clusters for larger datasets.

---

## 13. Pitfalls

| Pitfall | Consequence | Fix |
|---|---|---|
| Mixing vectors from different models | Garbage similarity | Store `model_name` with every vector; re-embed on model change |
| Forgetting query/document prefixes or `input_type` | Silent quality loss | Read model card; encode in your `Embedder` class |
| Not normalizing when using dot product | Longer texts dominate | Normalize at write time |
| English-only model on Khmer text | Poor retrieval | Multilingual model + evaluation |
| Chunks too large | Diluted meaning, missed facts | Tune chunk size with metrics |
| Chunks with no context | Ambiguous matches | Contextual headers |
| No threshold | Irrelevant results for off-topic queries | Calibrated minimum score |
| Pure vector search for IDs/codes | Misses exact matches | Hybrid search |
| Re-embedding unchanged documents | Wasted money | Hash content; embed only when hash changes |
| Assuming leaderboard = your data | Wrong model choice | Build your own eval set |

---

## 14. Exercises

1. **Implement from scratch.** Write `cosine_similarity_matrix(A, B)` for matrices of shape `(n, d)` and `(m, d)` using only NumPy, without normalizing in place. Verify against scikit-learn's `cosine_similarity`.

2. **Model bake-off.** Create a 40-query eval set for a domain you care about (e.g., RUPP course FAQ, or your Lost & Found app item descriptions). Include at least 10 Khmer queries. Compare recall@5 and MRR for: `all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, one E5 or BGE multilingual model, and one API model. Write a short recommendation.

3. **Chunk size experiment.** Take a long document (e.g., a course syllabus or a Spring Boot guide). Chunk at 300, 600, and 1200 characters with 15% overlap. Write 20 questions whose answers are in specific passages. Measure hit rate@3 for each chunk size.

4. **Contextual headers.** Repeat exercise 3 with and without `contextualize()`. Did it help?

5. **Embedding cache.** Build an `EmbeddingCache` wrapper that stores vectors keyed by `sha256(model_name + text)` in a local SQLite file, so re-running your scripts never re-embeds identical text.

6. **Lost & Found matching.** Given 50 "lost item" descriptions and 50 "found item" descriptions, use embeddings to propose the top-3 matches for each lost item. Evaluate on pairs you label by hand. What fails? (Colors? Brands? Locations?) What would you add?

7. **Matryoshka test.** Using a model that supports it, measure how recall@5 changes when truncating vectors to 50%, 25%, and 12.5% of their dimensions. Plot quality vs. storage.

---

## 15. Checklist

- [ ] You can explain embeddings as "meaning → geometry" and why dimensions aren't individually interpretable
- [ ] You know why vectors from different models can't be compared
- [ ] You can compute cosine similarity, dot product, and L2 distance, and know they rank identically for normalized vectors
- [ ] You can estimate storage for N vectors of dimension d
- [ ] You can embed text locally and via API, handling query/document asymmetry
- [ ] You built semantic search and understand its weaknesses (exact matches, thresholds)
- [ ] You can chunk documents with overlap and contextual headers
- [ ] You can measure recall@k, precision@k, hit rate, and MRR on a labeled set
- [ ] You choose embedding models by evaluating on your own data and language

**Next: Lesson 4 — Vector databases with PostgreSQL + pgvector.** NumPy in memory doesn't survive restarts, doesn't scale, and can't filter by metadata. We'll move everything into Postgres — the database you already use with Spring Boot.
