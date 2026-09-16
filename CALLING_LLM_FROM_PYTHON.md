# Lesson 2 — Calling an LLM from Python

> **Goal of this lesson:** Go from zero to writing production-quality LLM client code: configuration, single calls, multi-turn chat, streaming, error handling, retries, concurrency, cost tracking, provider abstraction, and a streaming HTTP API. By the end you'll have a reusable `llm` module you'll import in every later lesson.

**Prerequisites:** Lesson 1. Python 3.11+. An Anthropic API key (console.anthropic.com). Optionally an OpenAI key and/or [Ollama](https://ollama.com) for local models.

---

## Table of Contents

1. [Project setup](#1-project-setup)
2. [Your first call](#2-your-first-call)
3. [Understanding the response object](#3-understanding-the-response-object)
4. [Request parameters that matter](#4-request-parameters-that-matter)
5. [Multi-turn conversations](#5-multi-turn-conversations)
6. [Streaming](#6-streaming)
7. [Errors, retries, and timeouts](#7-errors-retries-and-timeouts)
8. [Async and concurrency](#8-async-and-concurrency)
9. [Token usage and cost tracking](#9-token-usage-and-cost-tracking)
10. [Other providers: OpenAI and local models](#10-other-providers-openai-and-local-models)
11. [A provider-agnostic client](#11-a-provider-agnostic-client)
12. [Serving an LLM over HTTP with FastAPI (streaming)](#12-serving-an-llm-over-http-with-fastapi-streaming)
13. [Testing LLM code](#13-testing-llm-code)
14. [Project: a CLI assistant](#14-project-a-cli-assistant)
15. [Exercises](#15-exercises)
16. [Checklist](#16-checklist)

---

## 1. Project setup

We'll use one repository for the entire course. Structure it like a real project from day one.

```
ai-course/
├── .env                  # secrets (NEVER commit)
├── .env.example          # template (commit this)
├── .gitignore
├── pyproject.toml
├── llm/                  # reusable module we build in this lesson
│   ├── __init__.py
│   ├── config.py
│   ├── client.py
│   └── costs.py
├── lesson02/
│   ├── first_call.py
│   ├── chat_cli.py
│   └── api.py
└── tests/
```

### 1.1 Environment

Using [`uv`](https://docs.astral.sh/uv/) (fast, modern) — or plain `venv` + `pip` if you prefer:

```bash
mkdir ai-course && cd ai-course
uv init
uv add anthropic openai python-dotenv pydantic-settings httpx fastapi "uvicorn[standard]" tenacity
uv add --dev pytest pytest-asyncio respx
```

Equivalent with pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install anthropic openai python-dotenv pydantic-settings httpx fastapi "uvicorn[standard]" tenacity pytest pytest-asyncio respx
```

### 1.2 Secrets

`.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_MODEL=claude-sonnet-5
LLM_FAST_MODEL=claude-haiku-4-5-20251001
```

`.gitignore`:

```
.env
.venv/
__pycache__/
```

> **Rule:** API keys never go in code, prompts, logs, or Git. Treat them like database passwords. In production, load them from a secret manager (AWS Secrets Manager, Vault, Kubernetes secrets).

### 1.3 Typed configuration

Just like `@ConfigurationProperties` in Spring Boot, centralize configuration with validation:

```python
# llm/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str | None = None

    llm_model: str = "claude-sonnet-5"
    llm_fast_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

settings = Settings()
```

If `ANTHROPIC_API_KEY` is missing, the app fails **at startup** with a clear error — much better than failing on the first user request.

> **Model names change.** Always check the provider's models page for current identifiers. Keeping model names in configuration (not hardcoded) lets you upgrade without code changes.

---

## 2. Your first call

```python
# lesson02/first_call.py
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=500,
    system="You are a senior backend engineer. Be concise and precise.",
    messages=[
        {"role": "user", "content": "In 3 bullet points, what is the difference between a JPA entity and a DTO?"}
    ],
)

print(response.content[0].text)
```

Run:

```bash
uv run python lesson02/first_call.py
```

That's the whole foundation. Everything else in this course — RAG, tools, agents — is built on this one call.

---

## 3. Understanding the response object

Print the raw response once so you know what you're working with:

```python
print(response.model_dump_json(indent=2))
```

```json
{
  "id": "msg_01A...",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "content": [
    { "type": "text", "text": "- **JPA entity**: ..." }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 38,
    "output_tokens": 112
  }
}
```

| Field | What to do with it |
|---|---|
| `id` | Log it. Useful when contacting support or tracing requests. |
| `content` | A **list of blocks**, not a string. Blocks can be `text`, `tool_use` (Lesson 7), `thinking`, etc. |
| `stop_reason` | **Always check.** `end_turn` = finished; `max_tokens` = truncated; `tool_use` = wants a tool; `stop_sequence` = hit your stop string. |
| `usage` | Tokens billed. Log for cost tracking (section 9). |

A safe text extractor (don't assume `content[0]` is text):

```python
def extract_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
```

Detect truncation:

```python
if response.stop_reason == "max_tokens":
    # Options: raise, retry with larger max_tokens, or ask the model to continue
    raise RuntimeError(f"Response truncated at {response.usage.output_tokens} tokens")
```

---

## 4. Request parameters that matter

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,              # REQUIRED. Upper bound on OUTPUT tokens.
    system="...",                 # instructions (string or list of blocks)
    messages=[...],               # alternating user/assistant
    temperature=0.2,              # 0–1. Lower = more focused.
    stop_sequences=["</answer>"], # stop when this string is generated
    metadata={"user_id": "hashed-user-123"},  # opaque id for abuse monitoring
)
```

### `max_tokens`
- Caps **output only**. Doesn't reserve or cost anything unless used.
- Set it to what you genuinely need plus a margin. Too low → truncation bugs. Too high → a runaway response can be slow and costly.

### `temperature`
- From Lesson 1: low for extraction/RAG/code, higher for creative tasks.

### `stop_sequences`
Useful to cut output at a delimiter:

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=300,
    system="Put your final answer inside <answer></answer> tags.",
    messages=[{"role": "user", "content": "What is 17 * 23? Think briefly, then answer."}],
    stop_sequences=["</answer>"],
)
text = extract_text(response)
answer = text.split("<answer>")[-1].strip()
print(response.stop_reason, "→", answer)   # stop_sequence → 391
```

### Content blocks: sending images

User content can also be a list of blocks. Example — reading a receipt image:

```python
import base64, pathlib

image_data = base64.standard_b64encode(pathlib.Path("receipt.jpg").read_bytes()).decode()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": "List the items and total on this receipt."},
        ],
    }],
)
print(extract_text(response))
```

---

## 5. Multi-turn conversations

The API is stateless (Lesson 1). You own the history.

```python
# lesson02/conversation.py
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

class Conversation:
    def __init__(self, system: str, model: str = "claude-sonnet-5", max_history_turns: int = 20):
        self.system = system
        self.model = model
        self.max_history_turns = max_history_turns
        self.messages: list[dict] = []

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            messages=self._window(),
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        self.messages.append({"role": "assistant", "content": text})
        return text

    def _window(self) -> list[dict]:
        # Keep the last N turns. Must start with a "user" message.
        window = self.messages[-self.max_history_turns * 2:]
        while window and window[0]["role"] != "user":
            window = window[1:]
        return window

conv = Conversation(system="You are a Spring Boot mentor. Keep answers under 120 words.")
print(conv.ask("What is @Transactional?"))
print(conv.ask("What happens if I call that method from the same class?"))  # relies on history
```

The second question ("that method") only works because we re-send the first exchange.

### Persisting history

In a real app, history lives in a database keyed by conversation ID:

```sql
CREATE TABLE chat_messages (
    id            BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content       JSONB NOT NULL,      -- store blocks, not just text (tools later!)
    input_tokens  INT,
    output_tokens INT,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chat_messages (conversation_id, created_at);
```

Store `content` as JSON blocks: once you add tool calling, messages contain structured blocks that must be replayed exactly.

---

## 6. Streaming

Without streaming, the user stares at a spinner until the *entire* response is generated. With streaming, text appears as soon as the first token is ready (TTFT from Lesson 1). For any user-facing chat, **stream by default**.

### 6.1 Simple text streaming

```python
# lesson02/stream.py
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-5",
    max_tokens=800,
    messages=[{"role": "user", "content": "Explain the Spring bean lifecycle step by step."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    final = stream.get_final_message()   # full message incl. usage + stop_reason

print("\n\n---")
print("stop_reason:", final.stop_reason)
print("usage:", final.usage)
```

### 6.2 Event-level streaming

Under the hood, the API sends Server-Sent Events. For fine-grained control (e.g., tool calls streaming in, measuring TTFT):

```python
import time

start = time.perf_counter()
first_token_at = None

with client.messages.stream(
    model="claude-sonnet-5",
    max_tokens=400,
    messages=[{"role": "user", "content": "Write a haiku about PostgreSQL."}],
) as stream:
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(event.delta.text, end="", flush=True)
        elif event.type == "message_stop":
            pass

end = time.perf_counter()
print(f"\n\nTTFT: {(first_token_at - start)*1000:.0f} ms | total: {(end - start)*1000:.0f} ms")
```

Event sequence you'll see: `message_start` → `content_block_start` → many `content_block_delta` → `content_block_stop` → `message_delta` (contains `stop_reason` and final usage) → `message_stop`.

---

## 7. Errors, retries, and timeouts

LLM APIs fail. Plan for it exactly as you would for any remote dependency.

### 7.1 Error types

| Status | Exception (Anthropic SDK) | Retry? | Typical cause |
|---|---|---|---|
| 400 | `BadRequestError` | No | Invalid params, too many tokens, malformed messages |
| 401 | `AuthenticationError` | No | Bad API key |
| 403 | `PermissionDeniedError` | No | Key lacks access |
| 404 | `NotFoundError` | No | Wrong model name |
| 413 | `RequestTooLargeError` | No | Payload too big |
| 429 | `RateLimitError` | **Yes, with backoff** | Too many requests/tokens per minute |
| 500 | `InternalServerError` | **Yes** | Provider issue |
| 529 | `APIStatusError` (overloaded) | **Yes** | Provider overloaded |
| — | `APIConnectionError` / `APITimeoutError` | **Yes** | Network problems |

### 7.2 Built-in retries

The official SDK already retries connection errors, 408, 409, 429, and 5xx with exponential backoff (2 retries by default). Configure it:

```python
client = anthropic.Anthropic(
    max_retries=4,
    timeout=60.0,   # seconds for the whole request
)

# Per-request override:
client.with_options(timeout=10.0, max_retries=1).messages.create(...)
```

### 7.3 Handling errors explicitly

```python
import logging
import anthropic

log = logging.getLogger("llm")

def safe_complete(prompt: str) -> str | None:
    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except anthropic.RateLimitError as e:
        log.warning("Rate limited after retries: %s", e)
        return None                          # degrade gracefully
    except anthropic.APIConnectionError as e:
        log.error("Network failure: %s", e)
        return None
    except anthropic.APIStatusError as e:
        # Non-retryable (4xx) or retries exhausted (5xx)
        log.error("API error %s: %s (request_id=%s)", e.status_code, e.message, e.response.headers.get("request-id"))
        raise
```

### 7.4 Custom retry policy with `tenacity`

When you need retries on *your own* conditions (e.g., invalid output, which we'll do in Lesson 6):

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

class TruncatedResponse(Exception):
    pass

@retry(
    retry=retry_if_exception_type((TruncatedResponse, anthropic.APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=20),
    reraise=True,
)
def complete_not_truncated(prompt: str, max_tokens: int = 500) -> str:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        raise TruncatedResponse()
    return extract_text(response)
```

> **Spring Boot analogy:** this is Resilience4j `@Retry` + `@TimeLimiter`. Add a circuit breaker and a fallback model at scale (Lesson 10).

---

## 8. Async and concurrency

Most AI backends do many LLM calls concurrently: batch classification, parallel retrieval + generation, many users at once. LLM calls are I/O-bound, so `asyncio` gives large speedups.

```python
# lesson02/async_batch.py
import asyncio
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.AsyncAnthropic()

REVIEWS = [
    "Delivery was fast and the vegetables were fresh.",
    "My order arrived two days late and the milk was spoiled.",
    "Okay prices, but the app keeps logging me out.",
    "Best mart in Phnom Penh, staff are very kind!",
    "Refund still not processed after a week.",
] * 4   # 20 reviews

SEMAPHORE = asyncio.Semaphore(5)   # max 5 concurrent requests → respect rate limits

async def classify(review: str) -> str:
    async with SEMAPHORE:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",     # small/fast model for simple tasks
            max_tokens=5,
            temperature=0,
            system="Classify sentiment. Reply with exactly one word: positive, negative, or mixed.",
            messages=[{"role": "user", "content": review}],
        )
        return response.content[0].text.strip().lower()

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(classify(r) for r in REVIEWS), return_exceptions=True)
    elapsed = time.perf_counter() - start

    for review, label in zip(REVIEWS[:5], results[:5]):
        print(f"{label!s:>10} | {review}")
    errors = [r for r in results if isinstance(r, Exception)]
    print(f"\n{len(REVIEWS)} calls in {elapsed:.1f}s, {len(errors)} errors")

asyncio.run(main())
```

Key points:
- **`Semaphore`** bounds concurrency. Unbounded `gather` over 10,000 items will hit rate limits instantly.
- **`return_exceptions=True`** stops one failure from cancelling the whole batch.
- **Model routing**: simple classification uses a small, fast, cheap model.

> For large offline jobs (thousands of requests, no latency need), use the provider's **Message Batches API** — it's asynchronous and significantly cheaper.

---

## 9. Token usage and cost tracking

Every professional AI system tracks cost per request, per feature, and per user. Start now.

```python
# llm/costs.py
from dataclasses import dataclass

# USD per 1M tokens. THESE ARE PLACEHOLDERS — copy current values from the provider's pricing page.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5":            {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00},
}

@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        price = PRICING.get(self.model)
        if price is None:
            return 0.0
        return (self.input_tokens * price["input"] + self.output_tokens * price["output"]) / 1_000_000

class CostTracker:
    def __init__(self):
        self.records: list[Usage] = []

    def add(self, model: str, usage) -> Usage:
        u = Usage(model, usage.input_tokens, usage.output_tokens)
        self.records.append(u)
        return u

    def summary(self) -> dict:
        return {
            "requests": len(self.records),
            "input_tokens": sum(r.input_tokens for r in self.records),
            "output_tokens": sum(r.output_tokens for r in self.records),
            "cost_usd": round(sum(r.cost_usd for r in self.records), 6),
        }
```

Usage:

```python
tracker = CostTracker()
response = client.messages.create(model="claude-sonnet-5", max_tokens=300,
                                  messages=[{"role": "user", "content": "Explain HikariCP in 2 sentences."}])
u = tracker.add(response.model, response.usage)
print(f"{u.input_tokens} in / {u.output_tokens} out → ${u.cost_usd:.6f}")
print(tracker.summary())
```

**Back-of-envelope estimation** (do this before building any feature):

```
Chatbot: 5,000 conversations/day × 6 turns × (3,000 input + 300 output tokens)
= 90M input + 9M output tokens per day
→ multiply by per-million prices → daily cost
```

You'll often discover that *input* tokens (re-sent history + retrieved docs + system prompt) dominate — which is why prompt caching and context trimming matter (Lesson 10).

---

## 10. Other providers: OpenAI and local models

A professional AI engineer is provider-flexible. The concepts are identical; the shapes differ slightly.

### 10.1 OpenAI

```python
from openai import OpenAI
client = OpenAI()   # reads OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4.1-mini",     # check OpenAI's docs for current model names
    max_tokens=300,
    temperature=0.2,
    messages=[
        {"role": "system", "content": "You are a concise backend mentor."},
        {"role": "user", "content": "What is connection pooling?"},
    ],
)
print(response.choices[0].message.content)
print(response.choices[0].finish_reason, response.usage)
```

| Concept | Anthropic Messages API | OpenAI Chat Completions |
|---|---|---|
| System prompt | top-level `system` | message with role `system`/`developer` |
| Output text | `response.content[i].text` (blocks) | `response.choices[0].message.content` |
| Stop reason | `stop_reason` (`end_turn`, `max_tokens`, `tool_use`) | `finish_reason` (`stop`, `length`, `tool_calls`) |
| Token usage | `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |
| `max_tokens` | required | optional |

OpenAI also offers a newer Responses API; the concepts transfer directly.

### 10.2 Local models with Ollama

Ollama exposes an **OpenAI-compatible** endpoint, so the OpenAI SDK works unchanged:

```bash
ollama pull llama3.2
ollama serve
```

```python
from openai import OpenAI

local = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is ignored
response = local.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Say hello in Khmer and English."}],
)
print(response.choices[0].message.content)
```

Local models are great for development, privacy-sensitive prototypes, and learning — typically weaker than frontier hosted models for complex reasoning and non-English languages.

---

## 11. A provider-agnostic client

Your application code should not be scattered with SDK-specific calls. Define an interface (like a Spring `@Service` interface with multiple implementations).

```python
# llm/client.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Protocol
import time
import logging

import anthropic
from openai import OpenAI

from llm.config import settings

log = logging.getLogger("llm")

@dataclass
class Message:
    role: str      # "user" | "assistant"
    content: str

@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float
    raw: object = field(repr=False, default=None)

class LLMClient(Protocol):
    def complete(self, messages: list[Message], system: str = "", *,
                 max_tokens: int = 1024, temperature: float = 0.3) -> Completion: ...
    def stream(self, messages: list[Message], system: str = "", *,
               max_tokens: int = 1024, temperature: float = 0.3) -> Iterator[str]: ...


class AnthropicClient:
    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def complete(self, messages, system="", *, max_tokens=1024, temperature=0.3) -> Completion:
        start = time.perf_counter()
        r = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        latency = (time.perf_counter() - start) * 1000
        text = "".join(b.text for b in r.content if b.type == "text")
        c = Completion(text, r.model, r.usage.input_tokens, r.usage.output_tokens,
                       r.stop_reason, latency, raw=r)
        log.info("llm.complete provider=anthropic model=%s in=%d out=%d stop=%s latency_ms=%.0f",
                 c.model, c.input_tokens, c.output_tokens, c.stop_reason, c.latency_ms)
        return c

    def stream(self, messages, system="", *, max_tokens=1024, temperature=0.3):
        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as s:
            yield from s.text_stream


class OpenAICompatibleClient:
    """Works for OpenAI and any OpenAI-compatible server (Ollama, vLLM)."""
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key or settings.openai_api_key)

    def _to_openai(self, messages, system):
        out = [{"role": "system", "content": system}] if system else []
        return out + [{"role": m.role, "content": m.content} for m in messages]

    def complete(self, messages, system="", *, max_tokens=1024, temperature=0.3) -> Completion:
        start = time.perf_counter()
        r = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=self._to_openai(messages, system),
        )
        latency = (time.perf_counter() - start) * 1000
        return Completion(
            text=r.choices[0].message.content or "",
            model=r.model,
            input_tokens=r.usage.prompt_tokens if r.usage else 0,
            output_tokens=r.usage.completion_tokens if r.usage else 0,
            stop_reason=r.choices[0].finish_reason,
            latency_ms=latency, raw=r,
        )

    def stream(self, messages, system="", *, max_tokens=1024, temperature=0.3):
        s = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=self._to_openai(messages, system), stream=True,
        )
        for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_client(provider: str = "anthropic", **kwargs) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openai":
        return OpenAICompatibleClient(**kwargs)
    if provider == "ollama":
        return OpenAICompatibleClient(base_url="http://localhost:11434/v1", api_key="ollama", **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
```

```python
# llm/__init__.py
from llm.client import get_client, Message, Completion, LLMClient
```

Usage anywhere in the course:

```python
from llm import get_client, Message

llm = get_client()
c = llm.complete([Message("user", "Name 3 PostgreSQL index types.")], system="Be brief.")
print(c.text, c.input_tokens, c.output_tokens, f"{c.latency_ms:.0f}ms")
```

> **Design note:** This abstraction intentionally covers only text. Tool calling and structured output have provider-specific details; in Lessons 6–8 we'll use SDKs directly first, then see how LangChain provides this abstraction for you — and what it costs you.

---

## 12. Serving an LLM over HTTP with FastAPI (streaming)

Your Spring Boot or React frontend will call a Python AI service over HTTP. Let's build that service with Server-Sent Events.

```python
# lesson02/api.py
import json
import uuid
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm import get_client, Message

app = FastAPI(title="AI Chat Service")
llm = get_client()

SYSTEM = "You are a helpful assistant for software engineering students. Be concise."
# Demo only — use Postgres/Redis in real apps.
CONVERSATIONS: dict[str, list[Message]] = defaultdict(list)

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    input_tokens: int
    output_tokens: int

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cid = req.conversation_id or str(uuid.uuid4())
    history = CONVERSATIONS[cid]
    history.append(Message("user", req.message))
    try:
        c = llm.complete(history[-20:], system=SYSTEM)
    except Exception as e:
        history.pop()
        raise HTTPException(status_code=502, detail="LLM provider error") from e
    history.append(Message("assistant", c.text))
    return ChatResponse(conversation_id=cid, reply=c.text,
                        input_tokens=c.input_tokens, output_tokens=c.output_tokens)

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    cid = req.conversation_id or str(uuid.uuid4())
    history = CONVERSATIONS[cid]
    history.append(Message("user", req.message))

    def event_stream():
        parts: list[str] = []
        yield f"event: meta\ndata: {json.dumps({'conversation_id': cid})}\n\n"
        try:
            for token in llm.stream(history[-20:], system=SYSTEM):
                parts.append(token)
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
            history.append(Message("assistant", "".join(parts)))
            yield "event: done\ndata: {}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': 'generation failed'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Run and test:

```bash
uv run uvicorn lesson02.api:app --reload
curl -N -X POST localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain REST vs gRPC in 4 bullets"}'
```

Consuming it from a React/Next.js frontend (you already know this side):

```typescript
async function streamChat(message: string, onToken: (t: string) => void) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const evt of events) {
      const [eventLine, dataLine] = evt.split("\n");
      if (eventLine === "event: token") onToken(JSON.parse(dataLine.slice(6)).text);
    }
  }
}
```

> **Architecture tip:** A common production layout is **React → Spring Boot (auth, business logic, persistence) → Python AI service (LLM orchestration)**. Alternatively, **Spring AI** lets you call LLMs directly from Java. Knowing both patterns makes you valuable on mixed teams.

---

## 13. Testing LLM code

You can't unit-test the model's intelligence (that's evaluation — Lesson 10), but you *must* unit-test your code around it: prompt building, parsing, error handling, history management. Mock the LLM.

```python
# tests/test_conversation.py
from llm.client import Completion, Message

class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, system="", **kwargs):
        self.calls.append({"messages": list(messages), "system": system})
        return Completion(self.replies.pop(0), "fake", 10, 5, "end_turn", 1.0)

    def stream(self, messages, system="", **kwargs):
        yield from self.complete(messages, system).text.split()

def test_history_is_resent_each_turn():
    fake = FakeLLM(["Hi!", "Your name is Heak."])
    history: list[Message] = []

    for user_text in ["My name is Heak", "What is my name?"]:
        history.append(Message("user", user_text))
        reply = fake.complete(history, system="test")
        history.append(Message("assistant", reply.text))

    second_call = fake.calls[1]["messages"]
    assert [m.content for m in second_call] == ["My name is Heak", "Hi!", "What is my name?"]
```

Testing the FastAPI endpoint with a dependency override is the same pattern as Spring's `@MockBean`.

---

## 14. Project: a CLI assistant

Combine everything: config, streaming, history, commands, cost tracking.

```python
# lesson02/chat_cli.py
import sys
import anthropic
from dotenv import load_dotenv

from llm.costs import CostTracker

load_dotenv()
client = anthropic.Anthropic(max_retries=3, timeout=60)
tracker = CostTracker()

SYSTEM = """You are a senior software engineering mentor.
- Prefer concrete examples in Java/Spring Boot or Python.
- Keep answers under 200 words unless asked for more.
- If you are not sure, say so."""

HELP = "Commands: /reset  /cost  /model <name>  /system <text>  /quit"

def main():
    model = "claude-sonnet-5"
    system = SYSTEM
    history: list[dict] = []
    print(f"CLI assistant ({model}). {HELP}")

    while True:
        try:
            user = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue

        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            if cmd == "/quit": break
            elif cmd == "/reset": history.clear(); print("history cleared")
            elif cmd == "/cost": print(tracker.summary())
            elif cmd == "/model" and arg: model = arg; print("model →", model)
            elif cmd == "/system" and arg: system = arg; print("system prompt updated")
            else: print(HELP)
            continue

        history.append({"role": "user", "content": user})
        print("ai  › ", end="", flush=True)
        try:
            with client.messages.stream(model=model, max_tokens=1024, system=system,
                                        messages=history[-30:]) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                final = stream.get_final_message()
        except anthropic.APIError as e:
            history.pop()
            print(f"\n[error] {e}", file=sys.stderr)
            continue

        reply = "".join(b.text for b in final.content if b.type == "text")
        history.append({"role": "assistant", "content": reply})
        u = tracker.add(model, final.usage)
        if final.stop_reason == "max_tokens":
            print("\n[warning] response truncated")
        print(f"\n  [{u.input_tokens} in / {u.output_tokens} out, ${u.cost_usd:.5f}]")

    print("\nSession:", tracker.summary())

if __name__ == "__main__":
    main()
```

Run it, have a 10-turn conversation, and watch `input_tokens` grow each turn. That growth is the context re-sending you learned about in Lesson 1 — now you've seen it with your own eyes.

---

## 15. Exercises

1. **Truncation handler.** Write `complete_with_continuation(prompt)` that, when `stop_reason == "max_tokens"`, appends the partial response as an assistant message and asks the model to continue, up to 3 times, then joins the parts.

2. **Token-budget history.** Replace the "last 30 messages" window in the CLI with a token budget (e.g., 8,000 input tokens) using `client.messages.count_tokens`. Drop oldest turns first.

3. **Summarizing memory.** When history exceeds 20 messages, summarize the oldest 10 into one short paragraph (using the fast model) and inject it into the system prompt as `<conversation_summary>`.

4. **Benchmark models.** Using async, send the same 20 prompts to a large and a small model. Record TTFT, total latency, output tokens, and cost. Make a table. When is the small model "good enough"?

5. **Adversarial test.** Take the RUPP course-assistant system prompt you wrote in Lesson 1's exercise 6. Run your 5 adversarial messages against it. Improve the prompt and rerun. Keep the prompts and results in a file — this is the seed of an evaluation set (Lesson 10).

6. **Spring Boot client.** Call your FastAPI `/chat` endpoint from a Spring Boot app using `RestClient`, with a timeout and a Resilience4j retry. Then call `/chat/stream` using `WebClient` and log each token.

7. **Provider swap.** Run the CLI against Ollama by switching to `get_client("ollama", model="llama3.2")`. Compare answer quality in Khmer.

---

## 16. Checklist

- [ ] API keys load from environment/config, never hardcoded
- [ ] You check `stop_reason` and handle `max_tokens` truncation
- [ ] You extract text from content *blocks* safely
- [ ] You can maintain multi-turn history and understand its token cost
- [ ] You stream responses and can measure TTFT
- [ ] You know which errors to retry and have timeouts configured
- [ ] You can run bounded concurrent requests with `asyncio` + `Semaphore`
- [ ] Every call logs model, tokens, latency, and cost
- [ ] You can call Anthropic, OpenAI, and a local model
- [ ] You have a FastAPI service with SSE streaming and tests with a fake LLM

**Next: Lesson 3 — Embeddings.** LLMs generate text from tokens; embeddings turn text into *meaning as numbers*, which is the foundation for semantic search and RAG.
