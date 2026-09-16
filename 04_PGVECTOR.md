# Lesson 4 — Vector Databases with PostgreSQL + pgvector

> **Goal of this lesson:** Store and search embeddings in PostgreSQL like a professional: schema design, bulk ingestion, exact and approximate (HNSW/IVFFlat) search, metadata filtering, hybrid keyword + vector search, performance tuning, and integration from both Python and Spring Boot. At the end you'll have a reusable `PgVectorStore` that Lesson 5's RAG system plugs straight into.

**Prerequisites:** Lesson 3. Docker. Basic SQL (you already use Postgres with Spring Boot).

```bash
uv add "psycopg[binary,pool]" pgvector numpy sentence-transformers
```

---

## Table of Contents

1. [Why a vector database — and why Postgres](#1-why-a-vector-database--and-why-postgres)
2. [Running Postgres with pgvector](#2-running-postgres-with-pgvector)
3. [pgvector fundamentals in pure SQL](#3-pgvector-fundamentals-in-pure-sql)
4. [Designing the schema](#4-designing-the-schema)
5. [Storing embeddings from Python](#5-storing-embeddings-from-python)
6. [Similarity search](#6-similarity-search)
7. [Indexes: exact vs. approximate search](#7-indexes-exact-vs-approximate-search)
8. [Metadata filtering](#8-metadata-filtering)
9. [Hybrid search (keyword + vector)](#9-hybrid-search-keyword--vector)
10. [A reusable PgVectorStore](#10-a-reusable-pgvectorstore)
11. [From Spring Boot](#11-from-spring-boot)
12. [Operations: migrations, re-embedding, scaling](#12-operations-migrations-re-embedding-scaling)
13. [Exercises](#13-exercises)
14. [Checklist](#14-checklist)

---

## 1. Why a vector database — and why Postgres

In Lesson 3 our "database" was a NumPy matrix. That breaks down fast:

| Need | NumPy in memory | Vector database |
|---|---|---|
| Survive restarts | ❌ | ✅ |
| Millions of vectors | RAM-bound, brute force | ANN indexes |
| Filter by tenant/category/date | Manual, slow | `WHERE` clauses |
| Concurrent writes and reads | ❌ | Transactions |
| Join with business data | ❌ | SQL joins |
| Backups, replication, access control | ❌ | ✅ |

### Options

| Type | Examples | When |
|---|---|---|
| Postgres extension | **pgvector** | You already run Postgres; tens of millions of vectors; need joins/transactions |
| Dedicated vector DB | Qdrant, Weaviate, Milvus, Pinecone | Very large scale, specialized features, separate team/infra |
| Search engines with vectors | Elasticsearch/OpenSearch | Heavy existing keyword search investment |
| Embedded/lightweight | Chroma, LanceDB, FAISS | Prototypes, local tools, offline pipelines |

**Why pgvector is the professional default for most teams:** one database, ACID transactions, your existing backups/monitoring/ORM, and the ability to write `WHERE tenant_id = ? AND status = 'published'` next to the vector search. You can move to a dedicated engine later if you actually hit its limits — most applications never do.

---

## 2. Running Postgres with pgvector

`docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    container_name: ai-course-db
    environment:
      POSTGRES_USER: ai
      POSTGRES_PASSWORD: ai
      POSTGRES_DB: ai_course
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    command: >
      postgres
        -c shared_buffers=512MB
        -c maintenance_work_mem=1GB
        -c max_parallel_maintenance_workers=4
volumes:
  pgdata:
```

```bash
docker compose up -d
docker exec -it ai-course-db psql -U ai -d ai_course
```

Add to `.env`:

```bash
DATABASE_URL=postgresql://ai:ai@localhost:5432/ai_course
```

Enable the extension (once per database):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

> Managed options (AWS RDS/Aurora, Google Cloud SQL, Azure, Supabase, Neon) support pgvector — check which version they offer, since newer versions add important features like iterative index scans.

---

## 3. pgvector fundamentals in pure SQL

Learn the primitives in `psql` before writing any Python.

### 3.1 The `vector` type

```sql
CREATE TABLE toy (
    id        SERIAL PRIMARY KEY,
    label     TEXT,
    embedding vector(3)          -- fixed dimension, enforced on insert
);

INSERT INTO toy (label, embedding) VALUES
  ('cat',      '[0.9, 0.1, 0.0]'),
  ('kitten',   '[0.85, 0.15, 0.05]'),
  ('dog',      '[0.7, 0.3, 0.1]'),
  ('car',      '[0.0, 0.2, 0.95]'),
  ('truck',    '[0.05, 0.25, 0.9]');

-- Wrong dimension → error
INSERT INTO toy (label, embedding) VALUES ('bad', '[1, 2]');
-- ERROR:  expected 3 dimensions, not 2
```

Related types: `halfvec` (16-bit floats, half the storage), `sparsevec` (sparse vectors), `bit` (binary vectors).

### 3.2 Distance operators

| Operator | Meaning | Index ops class |
|---|---|---|
| `<->` | Euclidean (L2) distance | `vector_l2_ops` |
| `<=>` | **Cosine distance** (1 − cosine similarity) | `vector_cosine_ops` |
| `<#>` | **Negative** inner product | `vector_ip_ops` |
| `<+>` | L1 (Manhattan) distance | `vector_l1_ops` |

All operators return a **distance: smaller = more similar**. That's why `<#>` is negated — so `ORDER BY ... ASC` works for every operator.

```sql
-- 3 nearest neighbours of "cat" by cosine distance
SELECT label,
       embedding <=> '[0.9, 0.1, 0.0]'          AS cosine_distance,
       1 - (embedding <=> '[0.9, 0.1, 0.0]')    AS cosine_similarity
FROM toy
ORDER BY embedding <=> '[0.9, 0.1, 0.0]'
LIMIT 3;
```

```
 label  | cosine_distance | cosine_similarity
--------+-----------------+-------------------
 cat    |               0 |                 1
 kitten |          0.0063 |            0.9937
 dog    |          0.0660 |            0.9340
```

### 3.3 Useful functions

```sql
SELECT vector_dims(embedding), vector_norm(embedding) FROM toy LIMIT 1;
SELECT l2_normalize('[3,4,0]'::vector);          -- [0.6,0.8,0]
SELECT AVG(embedding) FROM toy WHERE label IN ('cat','kitten');   -- centroid
SELECT subvector('[1,2,3,4]'::vector, 1, 2);     -- [1,2] (Matryoshka truncation)
```

`AVG(embedding)` is handy for recommendations: "average the vectors of products this user bought, find products near that centroid."

---

## 4. Designing the schema

A production-grade schema for documents and chunks, used in Lesson 5:

```sql
-- migrations/V1__vector_schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT        NOT NULL DEFAULT 'default',
    source_uri   TEXT        NOT NULL,              -- file path, URL, S3 key
    title        TEXT,
    content_hash TEXT        NOT NULL,              -- detect changes → avoid re-embedding
    metadata     JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_uri)
);

CREATE TABLE chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       TEXT        NOT NULL,
    chunk_index     INT         NOT NULL,
    content         TEXT        NOT NULL,
    token_count     INT,
    metadata        JSONB       NOT NULL DEFAULT '{}',   -- page, section, heading, language
    embedding       vector(384) NOT NULL,
    embedding_model TEXT        NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Vector index (section 7)
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Filters and keyword search
CREATE INDEX chunks_tenant_idx   ON chunks (tenant_id);
CREATE INDEX chunks_document_idx ON chunks (document_id);
CREATE INDEX chunks_metadata_gin ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX chunks_tsv_gin      ON chunks USING gin (tsv);
```

### Design decisions explained

| Decision | Why |
|---|---|
| Separate `documents` and `chunks` | One document → many chunks; delete/update a document atomically with `ON DELETE CASCADE` |
| `tenant_id` denormalized onto `chunks` | Filter vector search without a join; enforce data isolation |
| `content_hash` | Skip re-embedding unchanged documents (saves money) |
| `embedding_model` column | Detect mixed-model data; support migrations between models |
| Store `content` next to the vector | Retrieval returns the text directly — no second lookup |
| `metadata JSONB` | Flexible filters (page, section, language, access level) without schema churn |
| Generated `tsv` column | Keyword search for hybrid retrieval, always in sync |
| `'simple'` text search config | Language-neutral: no English stemming that would mangle other languages |

> **Khmer note:** Khmer is written without spaces between words, so Postgres full-text search can't split it into words. For Khmer keyword matching consider `pg_trgm` trigram indexes, or pre-segmenting Khmer text into words with a segmentation library before storing it in a separate search column.

---

## 5. Storing embeddings from Python

### 5.1 Connecting with psycopg 3 and pgvector

```python
# lesson04/db.py
import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)          # lets you pass/receive numpy arrays as vectors
    return conn
```

### 5.2 Ingesting documents

```python
# lesson04/ingest.py
import hashlib
import json
from psycopg.types.json import Jsonb

from lesson04.db import connect
from llm.embeddings import LocalEmbedder
from chunking import recursive_chunk      # from Lesson 3

embedder = LocalEmbedder()                # 384 dims — must match vector(384)

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ingest_document(conn, *, source_uri: str, title: str, text: str,
                    tenant_id: str = "default", metadata: dict | None = None) -> int | None:
    metadata = metadata or {}
    content_hash = sha256(text)

    with conn.transaction():
        existing = conn.execute(
            "SELECT id, content_hash FROM documents WHERE tenant_id=%s AND source_uri=%s",
            (tenant_id, source_uri),
        ).fetchone()

        if existing and existing[1] == content_hash:
            print(f"skip (unchanged): {source_uri}")
            return existing[0]

        if existing:   # changed → delete old chunks, update doc
            doc_id = existing[0]
            conn.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))
            conn.execute(
                "UPDATE documents SET title=%s, content_hash=%s, metadata=%s, updated_at=now() WHERE id=%s",
                (title, content_hash, Jsonb(metadata), doc_id),
            )
        else:
            doc_id = conn.execute(
                """INSERT INTO documents (tenant_id, source_uri, title, content_hash, metadata)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (tenant_id, source_uri, title, content_hash, Jsonb(metadata)),
            ).fetchone()[0]

        chunks = recursive_chunk(text, max_chars=900, overlap=120)
        texts_to_embed = [f"Document: {title}\n\n{c}" for c in chunks]   # contextual header
        vectors = embedder.embed_documents(texts_to_embed)

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, tenant_id, chunk_index, content, metadata,
                                       embedding, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, tenant_id, i, chunk, Jsonb({**metadata, "chunk_chars": len(chunk)}),
                     vec, embedder.name)
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ],
            )
    print(f"ingested {len(chunks)} chunks: {source_uri}")
    return doc_id

if __name__ == "__main__":
    conn = connect()
    ingest_document(
        conn,
        source_uri="policies/returns.md",
        title="KhmerMart Return Policy",
        text=open("data/returns.md", encoding="utf-8").read(),
        metadata={"category": "policy", "language": "en"},
    )
```

Key professional habits in this code:
- **One transaction** per document: either all chunks are replaced or none are.
- **Idempotent**: re-running ingestion on unchanged files costs nothing.
- **Model name stored** on every chunk.

### 5.3 Bulk loading with COPY (for large datasets)

`executemany` is fine for thousands of rows. For millions, use binary `COPY`, which is dramatically faster:

```python
def bulk_copy_chunks(conn, rows):
    """rows: iterable of (document_id, tenant_id, chunk_index, content, embedding(np.ndarray), model)"""
    with conn.cursor() as cur:
        with cur.copy(
            "COPY chunks (document_id, tenant_id, chunk_index, content, embedding, embedding_model) "
            "FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int8", "text", "int4", "text", "vector", "text"])
            for row in rows:
                copy.write_row(row)
```

**Tip:** for a huge initial load, create the HNSW index **after** loading the data — building it once is much faster than updating it on every insert.

---

## 6. Similarity search

```python
# lesson04/search.py
from lesson04.db import connect
from llm.embeddings import LocalEmbedder

embedder = LocalEmbedder()
conn = connect()

def search(query: str, k: int = 5, tenant_id: str = "default"):
    q = embedder.embed_query(query)
    return conn.execute(
        """
        SELECT c.id, d.title, c.chunk_index, c.content,
               1 - (c.embedding <=> %(q)s) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.tenant_id = %(tenant)s
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": q, "tenant": tenant_id, "k": k},
    ).fetchall()

for row in search("Can I return an opened product?"):
    cid, title, idx, content, sim = row
    print(f"[{sim:.3f}] {title} #{idx}: {content[:100]}...")
```

### Rules for queries that use the index

1. **`ORDER BY <distance expression> ASC LIMIT k`** — this exact shape lets the planner use the vector index.
2. The operator must match the index's ops class (`<=>` ↔ `vector_cosine_ops`).
3. Ordering by a computed similarity (`ORDER BY 1 - (embedding <=> q) DESC`) **won't use the index**. Order by the distance; compute similarity only in the `SELECT` list.

### Threshold filtering

```sql
SELECT id, content, 1 - (embedding <=> %(q)s) AS similarity
FROM chunks
WHERE embedding <=> %(q)s < 0.6          -- cosine distance threshold
ORDER BY embedding <=> %(q)s
LIMIT 10;
```

---

## 7. Indexes: exact vs. approximate search

### 7.1 Exact (no index)

Without a vector index, Postgres computes the distance to **every row**: perfect recall, linear cost. Fine for up to roughly tens of thousands of vectors.

### 7.2 Approximate Nearest Neighbor (ANN)

ANN indexes trade a small amount of **recall** (occasionally missing a true nearest neighbor) for huge speedups.

#### HNSW (Hierarchical Navigable Small World) — the default choice

A multi-layer graph: sparse "highway" layers on top for long jumps, dense layers at the bottom for precise local search.

```
Layer 2:   A ─────────────── F
Layer 1:   A ──── C ──── F ──── H
Layer 0:   A─B─C─D─E─F─G─H─I─J    (all vectors)
Search: start at top, greedily move toward the query, drop a layer, repeat.
```

```sql
CREATE INDEX chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Query-time knob (per session / transaction):
SET hnsw.ef_search = 100;
```

| Parameter | Default | Effect of increasing |
|---|---|---|
| `m` | 16 | More links per node → better recall, more memory, slower build |
| `ef_construction` | 64 | Better graph quality → better recall, slower build |
| `hnsw.ef_search` | 40 | Better recall per query, slower queries. **Also caps how many rows a query can return** — `LIMIT 100` with `ef_search = 40` can return at most 40 rows. |

Pros: excellent speed/recall trade-off, can be built on an empty table, handles inserts well.
Cons: slower to build, more memory.

#### IVFFlat

Clusters vectors into `lists` using k-means; at query time searches only the `probes` nearest clusters.

```sql
-- Build AFTER loading representative data (clusters are learned from existing rows)
CREATE INDEX chunks_embedding_ivf ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);          -- rule of thumb: rows/1000 up to 1M rows, sqrt(rows) beyond

SET ivfflat.probes = 32;      -- rule of thumb start: sqrt(lists)
```

Pros: faster build, less memory. Cons: generally lower recall at the same speed; quality degrades as data drifts from the original clusters (rebuild periodically).

**Recommendation:** start with HNSW. Choose IVFFlat only when build time or memory is a real constraint.

### 7.3 Dimension limits

Vector indexes on `vector` support a limited number of dimensions (2,000 at the time of writing); `halfvec` indexes support more (4,000). If your model outputs 3,072 dims, use `halfvec`, or shorten with a Matryoshka-capable model:

```sql
CREATE INDEX ON chunks USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);
-- the query must use the same expression:
-- ORDER BY embedding::halfvec(3072) <=> %(q)s::halfvec(3072)
```

### 7.4 Verify the index is used

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT id FROM chunks
ORDER BY embedding <=> (SELECT embedding FROM chunks WHERE id = 1)
LIMIT 5;
```

Look for `Index Scan using chunks_embedding_hnsw`. If you see `Seq Scan` + `Sort`, check: operator/ops-class mismatch, ordering by an expression, missing `LIMIT`, or a tiny table where the planner prefers a seq scan.

### 7.5 Measure ANN recall yourself

Professional teams verify the index quality instead of trusting defaults:

```python
# lesson04/ann_recall.py
import numpy as np
from lesson04.db import connect

conn = connect()
sample_ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY random() LIMIT 50")]

def neighbours(q, k, exact: bool, ef: int = 40):
    with conn.transaction():
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")   # force brute force
        else:
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
        rows = conn.execute("SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (q, k)).fetchall()
    return {r[0] for r in rows}

for ef in (10, 40, 100, 200):
    recalls = []
    for sid in sample_ids:
        q = conn.execute("SELECT embedding FROM chunks WHERE id=%s", (sid,)).fetchone()[0]
        truth = neighbours(q, 10, exact=True)
        approx = neighbours(q, 10, exact=False, ef=ef)
        recalls.append(len(truth & approx) / 10)
    print(f"ef_search={ef:>3}  recall@10={np.mean(recalls):.3f}")
```

### 7.6 Build performance tips

```sql
SET maintenance_work_mem = '2GB';               -- graph should fit in memory while building
SET max_parallel_maintenance_workers = 7;       -- parallel HNSW builds
CREATE INDEX CONCURRENTLY chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
```

`CONCURRENTLY` avoids locking writes on a live production table (slower, but safe).

---

## 8. Metadata filtering

Real queries almost always have filters: tenant, language, category, access level, date.

```sql
SELECT id, content
FROM chunks
WHERE tenant_id = 'mart-1'
  AND metadata @> '{"category": "policy", "language": "en"}'
ORDER BY embedding <=> %(q)s
LIMIT 5;
```

### 8.1 The filtering problem with ANN indexes

An HNSW scan finds the `ef_search` nearest candidates **first**, then applies `WHERE`. If only 1% of rows match the filter, most candidates get discarded and you might get **fewer than 5 results** — or none.

### 8.2 Solutions

**1. Iterative index scans (pgvector 0.8+)** — the scan keeps going until enough rows pass the filter:

```sql
SET hnsw.iterative_scan = relaxed_order;   -- or strict_order
SET hnsw.max_scan_tuples = 20000;          -- safety cap
```

**2. Partial indexes** for a small number of important, stable filter values:

```sql
CREATE INDEX chunks_policy_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
WHERE metadata->>'category' = 'policy';
```

**3. Partitioning** by tenant or category (each partition gets its own index):

```sql
CREATE TABLE chunks_p (LIKE chunks INCLUDING DEFAULTS) PARTITION BY LIST (tenant_id);
CREATE TABLE chunks_mart1 PARTITION OF chunks_p FOR VALUES IN ('mart-1');
```

**4. Exact search for highly selective filters** — if a B-tree filter narrows the table to a few thousand rows, brute-force distance over those rows is fast and has perfect recall. The planner often chooses this automatically when statistics show high selectivity.

### 8.3 Security: filtering is authorization

In a multi-tenant RAG app, the tenant/access filter is a **security boundary**. A missing `WHERE tenant_id = ...` leaks one customer's documents into another's answers. Enforce it in one repository method (never ad hoc), or use Postgres **Row-Level Security**:

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id'));
-- per request: SET app.tenant_id = 'mart-1';
```

---

## 9. Hybrid search (keyword + vector)

Recall from Lesson 3: vectors understand meaning but miss exact tokens (SKUs, error codes, names); keyword search is the opposite. Combine them.

### 9.1 Reciprocal Rank Fusion (RRF)

RRF merges ranked lists using ranks only (scores from different systems aren't comparable):

$$\text{RRF}(d) = \sum_{\text{lists}} \frac{1}{k + \text{rank}(d)} \quad (k \approx 60)$$

### 9.2 Hybrid search in one SQL query

```sql
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rank
    FROM chunks
    WHERE tenant_id = %(tenant)s
    ORDER BY embedding <=> %(q)s
    LIMIT 50
),
keyword AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
    FROM chunks c, websearch_to_tsquery('simple', %(text)s) AS query
    WHERE c.tenant_id = %(tenant)s AND c.tsv @@ query
    ORDER BY ts_rank_cd(c.tsv, query) DESC
    LIMIT 50
),
fused AS (
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score,
           s.rank AS semantic_rank,
           k.rank AS keyword_rank
    FROM semantic s
    FULL OUTER JOIN keyword k ON s.id = k.id
)
SELECT c.id, c.content, c.metadata, f.rrf_score, f.semantic_rank, f.keyword_rank
FROM fused f
JOIN chunks c ON c.id = f.id
ORDER BY f.rrf_score DESC
LIMIT %(k)s;
```

Returning `semantic_rank` and `keyword_rank` is extremely useful for debugging *why* a chunk was retrieved.

### 9.3 Weighted fusion

```sql
-- weight semantic results 1.0 and keyword results 0.5
0.5 * COALESCE(1.0 / (60 + k.rank), 0) + 1.0 * COALESCE(1.0 / (60 + s.rank), 0)
```

Tune weights with the evaluation harness from Lesson 3 — for example, for queries containing codes, keyword weight matters more.

### 9.4 Trigram search (works for any script, including Khmer)

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX chunks_content_trgm ON chunks USING gin (content gin_trgm_ops);

SELECT id, content, similarity(content, %(text)s) AS sim
FROM chunks
WHERE content %% %(text)s          -- %% escapes % for psycopg
ORDER BY sim DESC
LIMIT 20;
```

---

## 10. A reusable PgVectorStore

Wrap it all in a repository class — the Python equivalent of a Spring Data repository.

```python
# llm/vectorstore.py
from __future__ import annotations
from dataclasses import dataclass, field
import os

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from llm.embeddings import Embedder

@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    title: str
    source_uri: str
    content: str
    metadata: dict
    score: float
    debug: dict = field(default_factory=dict)

class PgVectorStore:
    def __init__(self, embedder: Embedder, dsn: str | None = None,
                 min_size: int = 1, max_size: int = 10):
        self.embedder = embedder
        self.pool = ConnectionPool(
            dsn or os.environ["DATABASE_URL"], min_size=min_size, max_size=max_size,
            configure=lambda conn: register_vector(conn), open=True,
        )

    # ---------- search ----------
    def similarity_search(self, query: str, *, k: int = 5, tenant_id: str = "default",
                          metadata_filter: dict | None = None, min_score: float | None = None,
                          ef_search: int = 100) -> list[SearchResult]:
        q = self.embedder.embed_query(query)
        sql = """
            SELECT c.id, c.document_id, d.title, d.source_uri, c.content, c.metadata,
                   1 - (c.embedding <=> %(q)s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.tenant_id = %(tenant)s
              AND c.embedding_model = %(model)s
              AND (%(filter)s::jsonb IS NULL OR c.metadata @> %(filter)s::jsonb)
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(k)s
        """
        params = {"q": q, "tenant": tenant_id, "model": self.embedder.name, "k": k,
                  "filter": Jsonb(metadata_filter) if metadata_filter else None}
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(sql, params).fetchall()
        results = [SearchResult(*r) for r in rows]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    def hybrid_search(self, query: str, *, k: int = 5, tenant_id: str = "default",
                      candidates: int = 50, semantic_weight: float = 1.0,
                      keyword_weight: float = 1.0, rrf_k: int = 60) -> list[SearchResult]:
        q = self.embedder.embed_query(query)
        sql = """
        WITH semantic AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rank
            FROM chunks
            WHERE tenant_id = %(tenant)s AND embedding_model = %(model)s
            ORDER BY embedding <=> %(q)s LIMIT %(n)s
        ),
        keyword AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
            FROM chunks c, websearch_to_tsquery('simple', %(text)s) query
            WHERE c.tenant_id = %(tenant)s AND c.tsv @@ query
            ORDER BY ts_rank_cd(c.tsv, query) DESC LIMIT %(n)s
        ),
        fused AS (
            SELECT COALESCE(s.id, kw.id) AS id,
                   %(sw)s * COALESCE(1.0 / (%(rrf_k)s + s.rank), 0)
                 + %(kw)s * COALESCE(1.0 / (%(rrf_k)s + kw.rank), 0) AS score,
                   s.rank AS s_rank, kw.rank AS k_rank
            FROM semantic s FULL OUTER JOIN keyword kw ON s.id = kw.id
        )
        SELECT c.id, c.document_id, d.title, d.source_uri, c.content, c.metadata,
               f.score, f.s_rank, f.k_rank
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC
        LIMIT %(k)s
        """
        params = {"q": q, "text": query, "tenant": tenant_id, "model": self.embedder.name,
                  "n": candidates, "k": k, "sw": semantic_weight, "kw": keyword_weight,
                  "rrf_k": rrf_k}
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [SearchResult(*r[:7], debug={"semantic_rank": r[7], "keyword_rank": r[8]})
                for r in rows]

    # ---------- write ----------
    def delete_document(self, source_uri: str, tenant_id: str = "default") -> int:
        with self.pool.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE tenant_id=%s AND source_uri=%s",
                               (tenant_id, source_uri))
            return cur.rowcount
```

Usage:

```python
from llm.embeddings import LocalEmbedder
from llm.vectorstore import PgVectorStore

store = PgVectorStore(LocalEmbedder())
for r in store.hybrid_search("return policy SKU-44821 opened box", k=5):
    print(f"{r.score:.4f} {r.debug} {r.title}: {r.content[:80]}")
```

---

## 11. From Spring Boot

You'll often keep ingestion/AI orchestration in Python but query the same tables from Java — or do everything in Java with Spring AI.

### 11.1 Plain JDBC with pgvector-java

```xml
<dependency>
    <groupId>com.pgvector</groupId>
    <artifactId>pgvector</artifactId>
    <version>0.1.6</version> <!-- check for the latest version -->
</dependency>
```

```java
// ChunkSearchRepository.java
@Repository
public class ChunkSearchRepository {

    private final JdbcClient jdbc;

    public ChunkSearchRepository(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    public record ChunkHit(long id, String title, String content, double similarity) {}

    public List<ChunkHit> search(float[] queryEmbedding, String tenantId, int k) {
        return jdbc.sql("""
                SELECT c.id, d.title, c.content,
                       1 - (c.embedding <=> :q) AS similarity
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.tenant_id = :tenant
                ORDER BY c.embedding <=> :q
                LIMIT :k
                """)
            .param("q", new PGvector(queryEmbedding))
            .param("tenant", tenantId)
            .param("k", k)
            .query((rs, i) -> new ChunkHit(
                rs.getLong("id"), rs.getString("title"),
                rs.getString("content"), rs.getDouble("similarity")))
            .list();
    }
}
```

The query embedding must come from **the same model** used at ingestion — e.g., Spring Boot calls the Python service's `/embed` endpoint, or both sides call the same embedding API.

### 11.2 Spring AI `VectorStore`

Spring AI provides a pgvector-backed `VectorStore` abstraction plus embedding clients:

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-starter-vector-store-pgvector</artifactId>
</dependency>
```

```yaml
spring:
  ai:
    vectorstore:
      pgvector:
        index-type: HNSW
        distance-type: COSINE_DISTANCE
        dimensions: 1024
        initialize-schema: true
```

```java
@Service
public class KnowledgeService {
    private final VectorStore vectorStore;

    public KnowledgeService(VectorStore vectorStore) { this.vectorStore = vectorStore; }

    public void add(String text, Map<String, Object> metadata) {
        vectorStore.add(List.of(new Document(text, metadata)));
    }

    public List<Document> search(String query) {
        return vectorStore.similaritySearch(
            SearchRequest.builder()
                .query(query)
                .topK(5)
                .similarityThreshold(0.5)
                .filterExpression("category == 'policy'")
                .build());
    }
}
```

Spring AI manages its own table schema. It's great for Java-first teams; the hand-written schema in this lesson gives more control (hybrid search, tenant columns, content hashes). Check the Spring AI reference docs for the current property names and embedding model configuration.

---

## 12. Operations: migrations, re-embedding, scaling

### 12.1 Migrations

Manage the schema with **Flyway** or **Liquibase** (Java) or **Alembic** (Python) — never by hand in production. `CREATE INDEX CONCURRENTLY` cannot run inside a transaction, so put it in its own migration configured to run non-transactionally.

### 12.2 Changing the embedding model (zero downtime)

1. Add a new column: `ALTER TABLE chunks ADD COLUMN embedding_v2 vector(1024);`
2. Backfill in batches with a background job (rate-limited, resumable).
3. Build the index on the new column `CONCURRENTLY`.
4. Evaluate retrieval quality on your eval set with the new column.
5. Switch reads behind a feature flag; then drop the old column.

### 12.3 Capacity planning

```
rows × (dims × 4 bytes + ~8 bytes header) ≈ raw vector storage
HNSW index ≈ often similar to or larger than the raw vectors
```

Keep the HNSW index in memory for low latency: check sizes with

```sql
SELECT pg_size_pretty(pg_relation_size('chunks_embedding_hnsw')) AS index_size,
       pg_size_pretty(pg_total_relation_size('chunks')) AS table_total;
```

### 12.4 Maintenance & monitoring

- **VACUUM/ANALYZE**: frequent updates/deletes leave dead tuples; autovacuum settings matter for write-heavy tables.
- **`pg_stat_statements`**: find slow similarity queries.
- **Latency budget**: vector query p95 is typically a small part of a RAG request; the LLM call dominates (Lesson 10).
- **Scaling reads**: read replicas for search traffic; partitioning for huge multi-tenant tables.

---

## 13. Exercises

1. **SQL fluency.** In `psql`, create the toy table, insert 10 hand-made 3-d vectors, and write queries for: top-3 by cosine, all rows within L2 distance 0.5, and the centroid of a label group. Explain why `<#>` is negative.

2. **Ingest a real corpus.** Download 50–200 Markdown/text documents (e.g., Spring Boot reference guide pages, or your course notes). Ingest them with the idempotent pipeline. Re-run and confirm nothing is re-embedded. Modify one file and confirm only that document is re-chunked.

3. **Index benchmark.** Generate 500k random normalized 384-d vectors with `COPY`. Measure p50/p95 query latency and recall@10 for: no index, HNSW (ef_search 20/40/100/200), IVFFlat (probes 1/10/40). Produce a recall-vs-latency table.

4. **Filter trap.** Tag 1% of rows with `{"category": "rare"}`. Query with that filter and `LIMIT 10` using HNSW with `iterative_scan = off`. How many rows come back? Turn on iterative scan and compare. Then try a partial index.

5. **Hybrid evaluation.** Build a 30-query eval set containing both natural-language questions and queries with exact codes (e.g., `ERR-4031`, product SKUs). Compare recall@5 for `similarity_search` vs. `hybrid_search`, and tune the weights.

6. **Row-Level Security.** Enable RLS on `chunks` with a tenant policy, create a non-superuser role for the app, and prove that a query without setting `app.tenant_id` returns zero rows.

7. **Spring Boot search endpoint.** Build `GET /api/search?q=...` in Spring Boot using `JdbcClient` + pgvector-java. Get the query embedding by calling a Python FastAPI `/embed` endpoint you write. Add a timeout and return results as JSON.

---

## 14. Checklist

- [ ] You can run pgvector in Docker and use `vector`, `halfvec`, and the distance operators
- [ ] You know `<=>` returns a *distance* and how to convert it to similarity
- [ ] Your schema separates documents/chunks and stores model name, content hash, metadata, and tenant
- [ ] Ingestion is transactional and idempotent
- [ ] You write queries in the `ORDER BY distance LIMIT k` shape and verify index usage with `EXPLAIN`
- [ ] You understand HNSW (`m`, `ef_construction`, `ef_search`) vs. IVFFlat (`lists`, `probes`) and can measure ANN recall
- [ ] You know why filters break ANN results and how to fix it (iterative scans, partial indexes, partitioning)
- [ ] You can implement hybrid search with RRF in SQL
- [ ] You treat tenant filtering as a security boundary
- [ ] You can query vectors from Spring Boot

**Next: Lesson 5 — RAG.** We now have all the parts: an LLM client (Lesson 2), embeddings (Lesson 3), and a vector store (Lesson 4). Time to build a real document Q&A system with citations, evaluation, and a web API.
