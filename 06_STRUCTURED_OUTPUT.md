# Lesson 6 — Structured Output

> **Goal of this lesson:** Make LLMs return data your code can trust — valid JSON that matches a schema, with correct types, enums, and business rules. You'll learn the full reliability ladder (prompting → validation & repair → tool-forcing → native constrained decoding), design schemas that make models *smarter*, and build reusable extraction, classification, and query-parsing components used by the rest of the course.

**Prerequisites:** Lessons 1–2 (logits, API calls), Lesson 5 (we'll fix its fragile JSON judge).

```bash
uv add anthropic openai pydantic tenacity
```

---

## Table of Contents

1. [Why structured output matters](#1-why-structured-output-matters)
2. [The reliability ladder](#2-the-reliability-ladder)
3. [Pydantic: your schema language](#3-pydantic-your-schema-language)
4. [Level 1: Prompting for JSON](#4-level-1-prompting-for-json)
5. [Level 2: Validate and repair](#5-level-2-validate-and-repair)
6. [Level 3: Forcing a tool call](#6-level-3-forcing-a-tool-call)
7. [Level 4: Native structured outputs (constrained decoding)](#7-level-4-native-structured-outputs-constrained-decoding)
8. [Schema design that improves accuracy](#8-schema-design-that-improves-accuracy)
9. [Business-rule validation](#9-business-rule-validation)
10. [Practical components](#10-practical-components)
11. [A reusable `structured()` helper](#11-a-reusable-structured-helper)
12. [Streaming, batching, and edge cases](#12-streaming-batching-and-edge-cases)
13. [Consuming structured output in Spring Boot](#13-consuming-structured-output-in-spring-boot)
14. [Testing and measuring reliability](#14-testing-and-measuring-reliability)
15. [Exercises](#15-exercises)
16. [Checklist](#16-checklist)

---

## 1. Why structured output matters

Free-form text is for humans. Software needs contracts.

```python
# Fragile: what could possibly go wrong?
reply = llm("Is this review positive or negative? 'Delivery was late but food was great'")
if reply == "positive": ...
```

Actual replies you'll get across thousands of calls:

```
"Positive"
"positive."
"Mixed — the delivery was late, but..."
"**Sentiment:** Positive"
"Here is the JSON: ```json {"sentiment": "positive"} ```"
```

Every AI feature that *does* something — writes to a database, calls an API, routes a ticket, fills a form, triggers a workflow — needs output conforming to a contract. Structured output is how you turn an LLM from a chatbot into a **component**.

Typical uses:
- **Extraction:** invoices, receipts, CVs, contracts → database rows
- **Classification & routing:** ticket category, priority, language, intent
- **Query understanding:** natural language → search filters or API parameters
- **Evaluation:** LLM-as-judge scores
- **Tool arguments:** the heart of Lesson 7
- **UI generation:** cards, forms, quiz questions rendered by React

---

## 2. The reliability ladder

| Level | Technique | Syntax valid? | Schema valid? | Business rules? | When |
|---|---|---|---|---|---|
| 1 | Prompt "return JSON" + parse | Usually | Often | No | Prototypes |
| 2 | + Pydantic validation + repair retries | Yes (after retries) | Yes (after retries) | Can check | Any provider, any model |
| 3 | Force a tool call with an input schema | Yes | Very likely | No | Broad provider support |
| 4 | Native structured outputs / constrained decoding | **Guaranteed** | **Guaranteed** (supported schema features) | No | Default for production when available |
| + | Business validation in your code | — | — | **Yes** | **Always** |

Key insight from Lesson 1: constrained decoding works by **masking logits** — at every step, tokens that would make the JSON invalid for the schema get probability zero. That's why Level 4 can *guarantee* structure, while Levels 1–3 can only make it likely.

**But no level guarantees the content is *correct*.** A schema guarantees `"total": 12.5` is a number, not that the receipt total is actually 12.5. That's what validation (section 9) and evaluation (Lesson 10) are for.

---

## 3. Pydantic: your schema language

Pydantic models are the Python equivalent of Java records + Bean Validation + Jackson — and they generate JSON Schema, which every LLM structured-output API consumes.

```python
# schemas/ticket.py
from typing import Literal
from pydantic import BaseModel, Field

class TicketTriage(BaseModel):
    """Triage result for a customer support message."""
    category: Literal["delivery", "payment", "product_quality", "account", "other"] = Field(
        description="Primary topic of the message")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="urgent = safety issue or money lost; high = order blocked; medium = inconvenience; low = question")
    language: Literal["km", "en", "other"]
    summary: str = Field(description="One sentence summary in English", max_length=200)
    order_id: str | None = Field(default=None, description="Order ID like 'KM-123456' if mentioned, else null")
    needs_human: bool = Field(description="True if an agent must handle it (refunds, complaints, legal)")

import json
print(json.dumps(TicketTriage.model_json_schema(), indent=2))
```

Output (abridged):

```json
{
  "title": "TicketTriage",
  "description": "Triage result for a customer support message.",
  "type": "object",
  "properties": {
    "category": {"enum": ["delivery", "payment", "product_quality", "account", "other"],
                 "type": "string", "description": "Primary topic of the message"},
    "order_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null,
                 "description": "Order ID like 'KM-123456' if mentioned, else null"},
    ...
  },
  "required": ["category", "priority", "language", "summary", "needs_human"]
}
```

Validation:

```python
from pydantic import ValidationError

try:
    TicketTriage.model_validate({"category": "shipping", "priority": "high", "language": "en",
                                 "summary": "Late order", "needs_human": False})
except ValidationError as e:
    print(e)
# category: Input should be 'delivery', 'payment', 'product_quality', 'account' or 'other'
```

That error message is exactly what we'll feed back to the model in Level 2.

---

## 4. Level 1: Prompting for JSON

```python
# level1_prompt.py
import json
import re
import anthropic
from schemas.ticket import TicketTriage

client = anthropic.Anthropic()

SYSTEM = f"""You triage customer support messages for KhmerMart.
Respond with ONLY a JSON object matching this JSON Schema. No markdown, no explanation.

{json.dumps(TicketTriage.model_json_schema())}"""

def extract_json(text: str) -> dict:
    """Tolerate code fences and surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])

message = "Bong, I paid with ABA twice for order KM-884512 and still no delivery!! Please refund"

r = client.messages.create(model="claude-sonnet-5", max_tokens=400, temperature=0,
                           system=SYSTEM, messages=[{"role": "user", "content": message}])
data = extract_json(r.content[0].text)
triage = TicketTriage.model_validate(data)
print(triage)
```

This works most of the time with strong models. "Most of the time" means that at 100,000 requests/day, you'll get a steady trickle of crashes. Move up the ladder.

---

## 5. Level 2: Validate and repair

When validation fails, send the model its own output plus the exact error, and ask it to fix it. This is provider-agnostic and works even with small local models.

```python
# level2_repair.py
from typing import TypeVar
import json
import anthropic
from pydantic import BaseModel, ValidationError

client = anthropic.Anthropic()
T = TypeVar("T", bound=BaseModel)

class StructuredOutputError(Exception):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts

def complete_validated(model_cls: type[T], system: str, user: str, *,
                       model: str = "claude-sonnet-5", max_attempts: int = 3) -> T:
    schema = json.dumps(model_cls.model_json_schema())
    full_system = f"{system}\n\nRespond with ONLY a JSON object matching this JSON Schema:\n{schema}"
    messages = [{"role": "user", "content": user}]
    attempts = []

    for attempt in range(1, max_attempts + 1):
        r = client.messages.create(model=model, max_tokens=1500, temperature=0,
                                   system=full_system, messages=messages)
        text = "".join(b.text for b in r.content if b.type == "text")
        attempts.append(text)

        if r.stop_reason == "max_tokens":
            error = "Your output was cut off because it was too long. Return a shorter, complete JSON object."
        else:
            try:
                return model_cls.model_validate(extract_json(text))
            except (ValueError, json.JSONDecodeError) as e:
                error = f"Your output was not valid JSON: {e}"
            except ValidationError as e:
                error = f"Your JSON did not match the schema:\n{e}"

        # Repair turn: show the model its output and the error
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"{error}\n\nReturn the corrected JSON object only."},
        ]

    raise StructuredOutputError(f"Failed after {max_attempts} attempts", attempts)
```

Log `attempt` counts in production. If more than ~1% of calls need repair, improve the prompt/schema or move up the ladder.

> **Library note:** [Instructor](https://python.useinstructor.com/) packages this pattern (Pydantic + retries + many providers). It's worth knowing, but build it once yourself first so you understand what it does.

---

## 6. Level 3: Forcing a tool call

Tool calling (Lesson 7) lets a model return **arguments as JSON** matching an `input_schema`. If you define one tool and **force** the model to call it, you get structured data — the model was specifically trained to produce schema-shaped tool arguments.

```python
# level3_tool_forcing.py
import anthropic
from schemas.ticket import TicketTriage

client = anthropic.Anthropic()

def triage(message: str) -> TicketTriage:
    r = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        temperature=0,
        system="You triage customer support messages for KhmerMart.",
        tools=[{
            "name": "record_triage",
            "description": "Record the triage result for the customer message.",
            "input_schema": TicketTriage.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "record_triage"},   # FORCE this tool
        messages=[{"role": "user", "content": message}],
    )
    block = next(b for b in r.content if b.type == "tool_use")
    return TicketTriage.model_validate(block.input)     # block.input is already a dict

print(triage("ខ្ញុំមិនអាចចូលគណនីបានទេ ពាក្យសម្ងាត់ខុស"))   # "I can't log into my account, wrong password"
```

Notes:
- `block.input` arrives as a parsed dict — no string parsing.
- **Still validate** with Pydantic: tool forcing makes schema conformance very likely, but without strict/constrained mode it's not a hard guarantee.
- Many providers support a **strict** flag on tool definitions that turns on constrained decoding for tool arguments. Check your provider's docs; with it, Level 3 becomes Level 4.

---

## 7. Level 4: Native structured outputs (constrained decoding)

Modern APIs can constrain generation to your JSON Schema directly.

### 7.1 Anthropic

The Python SDK's `messages.parse()` accepts a Pydantic model, sends the schema, and returns a parsed object:

```python
# level4_native_anthropic.py
import anthropic
from schemas.ticket import TicketTriage

client = anthropic.Anthropic()

response = client.messages.parse(
    model="claude-sonnet-5",
    max_tokens=600,
    system="You triage customer support messages for KhmerMart.",
    messages=[{"role": "user", "content": "The rice bag arrived torn and half empty. Order KM-102938."}],
    output_format=TicketTriage,
)
triage: TicketTriage = response.parsed_output
print(triage.category, triage.priority, triage.order_id)
```

Using a raw JSON Schema instead of Pydantic (e.g., schema loaded from a file or shared with Java):

```python
from anthropic import transform_schema

schema = transform_schema(TicketTriage.model_json_schema())   # adapts unsupported schema features
r = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=600,
    messages=[{"role": "user", "content": "..."}],
    output_config={"format": {"type": "json_schema", "schema": schema}},
)
data = TicketTriage.model_validate_json(r.content[0].text)
```

### 7.2 OpenAI

```python
from openai import OpenAI
from schemas.ticket import TicketTriage

client = OpenAI()
completion = client.chat.completions.parse(
    model="gpt-4.1-mini",     # check docs for current models
    messages=[
        {"role": "system", "content": "You triage customer support messages for KhmerMart."},
        {"role": "user", "content": "The rice bag arrived torn and half empty. Order KM-102938."},
    ],
    response_format=TicketTriage,
)
msg = completion.choices[0].message
if msg.refusal:
    print("refused:", msg.refusal)
else:
    print(msg.parsed)
```

### 7.3 Local models (Ollama)

Ollama accepts a JSON Schema in its `format` field:

```python
import httpx
from schemas.ticket import TicketTriage

r = httpx.post("http://localhost:11434/api/chat", timeout=120, json={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Triage: 'Where is my order KM-555111? It's 3 days late.'"}],
    "format": TicketTriage.model_json_schema(),
    "stream": False,
    "options": {"temperature": 0},
})
print(TicketTriage.model_validate_json(r.json()["message"]["content"]))
```

### 7.4 Limitations to know

- **Supported schema subset:** constrained decoding supports most common JSON Schema features, but some constraints (e.g., certain numeric ranges, string lengths, complex patterns, recursive schemas) may be unsupported or stripped by SDK transforms. Read the docs' "supported features" list. **Keep validating with Pydantic** so dropped constraints are still enforced.
- **First-request latency:** providers compile the schema into a grammar; the first request with a new schema can be slower (compiled grammars are cached).
- **Truncation still possible:** if `max_tokens` is too low, you get incomplete output — check `stop_reason`.
- **Refusals:** a model may refuse unsafe requests; handle that path explicitly.
- **Guaranteed structure ≠ correct content.** Hallucinated values are perfectly schema-valid.

---

## 8. Schema design that improves accuracy

The schema is part of the prompt. Design it like an API contract **and** like instructions.

### 8.1 Descriptions are instructions

```python
class Bad(BaseModel):
    p: str
    d: str

class Good(BaseModel):
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="urgent = customer safety or money already lost; high = order cannot be completed; "
                    "medium = inconvenience; low = general question")
    delivery_date: str | None = Field(
        description="ISO date YYYY-MM-DD of the promised delivery if explicitly stated; null if not stated")
```

### 8.2 Enums over free strings

`Literal[...]` eliminates spelling variants and makes routing code trivial. Always include an escape hatch (`"other"`, `"unknown"`) so the model isn't forced to pick a wrong category.

### 8.3 Nullable instead of guessing

If a field may be absent, make it `X | None` and **say** "null if not stated." Required fields with no correct value force the model to invent one — schemas can *cause* hallucinations.

### 8.4 Reasoning before answers (field order matters)

Models generate fields **in order**. Put a reasoning field *before* the decision so the model thinks first:

```python
class RefundDecision(BaseModel):
    reasoning: str = Field(description="Step-by-step: check policy conditions against the facts, max 80 words")
    eligible: bool
    policy_clause: str | None = Field(description="Exact clause identifier from the policy, e.g. '3.2'")
```

If `eligible` came first, the reasoning would merely justify a decision already made. (Don't show `reasoning` to end users unless intended — log it for debugging.)

### 8.5 Evidence fields reduce hallucination

```python
class ExtractedFact(BaseModel):
    value: str
    evidence: str = Field(description="Verbatim quote from the input that supports the value")
```

You can then verify `evidence in source_text` in code — a cheap, powerful hallucination check.

### 8.6 Keep it flat and focused

- Avoid deeply nested schemas; 2–3 levels is plenty.
- Ask for **one job per call**. "Extract 40 fields and also classify and also summarize" degrades all three. Split into calls, possibly parallel.
- Name fields in plain English (`customer_phone`, not `cph`).

### 8.7 Lists with bounds

```python
class Keywords(BaseModel):
    keywords: list[str] = Field(description="3 to 8 search keywords, most important first",
                                min_length=1, max_length=8)
```

---

## 9. Business-rule validation

Schemas check shape. Your domain has rules. Encode them as Pydantic validators — they run on every LLM output.

```python
# schemas/receipt.py
from datetime import date
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class LineItem(BaseModel):
    name: str
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    line_total: Decimal = Field(ge=0)

class Receipt(BaseModel):
    merchant_name: str
    purchase_date: date | None = Field(description="Date of purchase, null if not visible")
    currency: Literal["USD", "KHR"]
    items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal

    @field_validator("purchase_date")
    @classmethod
    def not_in_future(cls, v):
        if v and v > date.today():
            raise ValueError("purchase_date cannot be in the future")
        return v

    @model_validator(mode="after")
    def totals_are_consistent(self):
        for item in self.items:
            expected = (item.quantity * item.unit_price).quantize(Decimal("0.01"))
            if abs(expected - item.line_total) > Decimal("0.05"):
                raise ValueError(f"line_total for '{item.name}' should be about {expected}, got {item.line_total}")
        items_sum = sum(i.line_total for i in self.items)
        tolerance = Decimal("0.05") if self.currency == "USD" else Decimal("500")
        base = self.subtotal if self.subtotal is not None else items_sum
        if abs(items_sum - base) > tolerance:
            raise ValueError(f"sum of line items ({items_sum}) does not match subtotal ({base})")
        if abs(base + (self.tax or 0) - self.total) > tolerance:
            raise ValueError(f"subtotal + tax ({base + (self.tax or 0)}) does not match total ({self.total})")
        return self
```

Combine with the repair loop: a validation error like *"sum of line items (12.50) does not match total (15.50)"* often prompts the model to re-read the receipt and catch a missed item. This is **self-correction driven by deterministic checks** — one of the most effective patterns in applied AI.

When validation *keeps* failing, don't guess: route to human review.

```python
from enum import Enum

class ExtractionStatus(str, Enum):
    OK = "ok"
    NEEDS_REVIEW = "needs_review"

def extract_receipt_or_review(image_b64: str):
    try:
        return ExtractionStatus.OK, extract_receipt(image_b64)   # repair loop inside
    except StructuredOutputError as e:
        return ExtractionStatus.NEEDS_REVIEW, e.attempts[-1]
```

---

## 10. Practical components

### 10.1 Receipt extraction from an image (vision + structured output)

```python
# components/receipt_extractor.py
import base64
from pathlib import Path
import anthropic
from pydantic import ValidationError
from schemas.receipt import Receipt

client = anthropic.Anthropic()

SYSTEM = """You extract data from photos of store receipts in Cambodia.
- Receipts may be in Khmer, English, or both. Prices may be in USD or KHR (riel).
- Copy numbers exactly as printed. Do not compute missing values unless clearly derivable.
- If a field is not visible, use null."""

def extract_receipt(path: str, max_attempts: int = 3) -> Receipt:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    media_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    messages = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
        {"type": "text", "text": "Extract this receipt."},
    ]}]

    for attempt in range(max_attempts):
        r = client.messages.parse(model="claude-sonnet-5", max_tokens=2000, system=SYSTEM,
                                  messages=messages, output_format=Receipt)
        # parse() guarantees JSON structure; our model validators enforce business rules.
        # Depending on SDK version, a validator failure may raise ValidationError here.
        try:
            return r.parsed_output if r.parsed_output else Receipt.model_validate_json(r.content[0].text)
        except ValidationError as e:
            messages += [
                {"role": "assistant", "content": r.content[0].text},
                {"role": "user", "content": f"The extraction failed validation:\n{e}\n"
                                            "Look at the image again carefully and return a corrected extraction."},
            ]
    raise RuntimeError("Receipt extraction needs human review")
```

> Because SDK parse helpers may run your Pydantic validators themselves, wrap the whole call in `try/except ValidationError` in real code and feed the error into the repair turn the same way.

### 10.2 Natural language → search filters (self-querying for RAG)

Upgrade Lesson 5: turn a question into a semantic query **plus** metadata filters.

```python
# components/query_parser.py
from typing import Literal
from pydantic import BaseModel, Field
import anthropic

client = anthropic.Anthropic()

class ProductSearch(BaseModel):
    semantic_query: str = Field(description="What the user is looking for, without price/brand/category constraints")
    category: Literal["groceries", "household", "electronics", "health", "beverages"] | None = None
    max_price_usd: float | None = Field(default=None, description="Upper price bound in USD if stated. Convert riel at 4000 KHR = 1 USD")
    brand: str | None = None
    in_stock_only: bool = Field(default=False)
    sort: Literal["relevance", "price_asc", "price_desc", "newest"] = "relevance"

def parse_search(user_query: str) -> ProductSearch:
    r = client.messages.parse(
        model="claude-haiku-4-5-20251001", max_tokens=300,
        system="Convert the shopper's request into structured search parameters. Only set filters the user actually stated.",
        messages=[{"role": "user", "content": user_query}],
        output_format=ProductSearch,
    )
    return r.parsed_output

p = parse_search("cheap rice cooker under 40000 riel that's available now, cheapest first")
# ProductSearch(semantic_query='rice cooker', category='electronics', max_price_usd=10.0,
#               brand=None, in_stock_only=True, sort='price_asc')
```

Then translate to SQL safely — **never** let the LLM write raw SQL for this:

```python
def to_sql(p: ProductSearch):
    where, params = ["1=1"], {"q": embed(p.semantic_query)}
    if p.category:       where.append("category = %(category)s"); params["category"] = p.category
    if p.max_price_usd:  where.append("price_usd <= %(max_price)s"); params["max_price"] = p.max_price_usd
    if p.brand:          where.append("brand ILIKE %(brand)s"); params["brand"] = p.brand
    if p.in_stock_only:  where.append("stock > 0")
    order = {"relevance": "embedding <=> %(q)s", "price_asc": "price_usd ASC",
             "price_desc": "price_usd DESC", "newest": "created_at DESC"}[p.sort]
    return f"SELECT * FROM products WHERE {' AND '.join(where)} ORDER BY {order} LIMIT 20", params
```

The LLM chooses from a **closed set of options** (enums); your code builds the query. This is the safe pattern for connecting LLMs to databases.

### 10.3 Fixing Lesson 5's judge

```python
from pydantic import BaseModel, Field

class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Brief justification, max 60 words, written BEFORE the scores")
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    abstained: bool

def judge(question, expected, sources_text, answer) -> JudgeVerdict:
    r = client.messages.parse(
        model="claude-sonnet-5", max_tokens=400, temperature=0,
        system="You are a strict evaluator of a question-answering system.",
        messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
            question=question, expected=expected, sources=sources_text, answer=answer)}],
        output_format=JudgeVerdict,
    )
    return r.parsed_output
```

No more `text.find("{")`.

### 10.4 Batch entity extraction

```python
class Person(BaseModel):
    name: str
    role: str | None = None
    organization: str | None = None

class People(BaseModel):
    people: list[Person] = Field(description="Every distinct person mentioned. Empty list if none.")
```

**Always wrap lists in an object** — a top-level object with a `people` field is more portable than a bare array and leaves room to add fields later.

---

## 11. A reusable `structured()` helper

One function for the whole codebase: native structured outputs when available, repair loop as fallback, consistent logging.

```python
# llm/structured.py
from __future__ import annotations
import json
import logging
import time
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm.structured")
T = TypeVar("T", bound=BaseModel)
_client = anthropic.Anthropic()

class StructuredOutputError(Exception):
    def __init__(self, message: str, last_output: str | None = None):
        super().__init__(message)
        self.last_output = last_output

def structured(model_cls: type[T], *, user: str | list, system: str = "",
               model: str = "claude-sonnet-5", max_tokens: int = 2000,
               max_repairs: int = 2, temperature: float = 0.0) -> T:
    messages: list[dict] = [{"role": "user", "content": user}]
    last_text = None
    start = time.perf_counter()

    for attempt in range(max_repairs + 1):
        try:
            r = _client.messages.parse(model=model, max_tokens=max_tokens, temperature=temperature,
                                       system=system or anthropic.NOT_GIVEN,
                                       messages=messages, output_format=model_cls)
            last_text = "".join(b.text for b in r.content if b.type == "text")
            if r.stop_reason == "max_tokens":
                raise StructuredOutputError("truncated", last_text)
            if r.stop_reason == "refusal":
                raise StructuredOutputError("model refused", last_text)
            result = r.parsed_output or model_cls.model_validate_json(last_text)
            log.info("structured ok schema=%s attempts=%d ms=%.0f in=%d out=%d",
                     model_cls.__name__, attempt + 1, (time.perf_counter() - start) * 1000,
                     r.usage.input_tokens, r.usage.output_tokens)
            return result
        except ValidationError as e:          # business rules (validators) failed
            log.warning("structured validation failed schema=%s attempt=%d: %s",
                        model_cls.__name__, attempt + 1, e.errors()[:3])
            if attempt == max_repairs:
                raise StructuredOutputError(f"validation failed: {e}", last_text) from e
            messages += [
                {"role": "assistant", "content": last_text or "{}"},
                {"role": "user", "content": f"That output failed validation:\n{e}\nFix the problems and try again."},
            ]
    raise StructuredOutputError("unreachable")
```

Every later lesson uses `structured(...)` whenever it needs typed LLM output.

---

## 12. Streaming, batching, and edge cases

### 12.1 Streaming partial JSON

For long structured outputs displayed live (e.g., a generated quiz appearing card by card), stream the JSON text and parse incrementally with a partial-JSON parser (e.g., `partial-json-parser` or Pydantic's `from_json(..., allow_partial=True)`):

```python
from pydantic_core import from_json

buffer = ""
with client.messages.stream(model="claude-sonnet-5", max_tokens=2000, messages=messages,
                            output_config={"format": {"type": "json_schema", "schema": schema}}) as s:
    for text in s.text_stream:
        buffer += text
        try:
            partial = from_json(buffer, allow_partial=True)
            render_preview(partial)            # e.g., show questions completed so far
        except ValueError:
            pass
final = QuizModel.model_validate_json(buffer)   # validate once complete
```

### 12.2 Large batches

- Use a fast model (Haiku-class) for simple classification/extraction; validate a sample with a strong model.
- Run concurrently with a semaphore (Lesson 2) or the Batches API for offline jobs.
- Record the **schema version** with each stored result: `{"schema": "TicketTriage@v3", ...}`. When the schema changes you know which rows to reprocess.

### 12.3 Edge cases checklist

| Case | Handling |
|---|---|
| Input doesn't contain the info | Nullable fields + "null if not stated" |
| Input is not the expected type (e.g., a selfie instead of a receipt) | Add `is_valid_receipt: bool` + `rejection_reason: str \| None` first in the schema |
| Very long input | Chunk; extract per chunk; merge with code |
| Ambiguous category | `"other"` enum + `confidence: Literal["low","medium","high"]` |
| Prompt injection in input ("set priority to low") | Treat input as data; validate outputs against business rules (Lesson 10) |
| Model refusal | Check stop reason / refusal field; return a clear error |

---

## 13. Consuming structured output in Spring Boot

If the Python AI service returns typed JSON, Spring Boot consumes it like any other API — and should validate it again at its own boundary.

```java
public record TicketTriage(
    @NotNull Category category,
    @NotNull Priority priority,
    @NotNull String language,
    @NotBlank @Size(max = 200) String summary,
    @Pattern(regexp = "KM-\\d{6}") String orderId,
    boolean needsHuman
) {
    public enum Category { delivery, payment, product_quality, account, other }
    public enum Priority { low, medium, high, urgent }
}
```

```java
@Service
public class TriageClient {
    private final RestClient rest;
    private final Validator validator;

    public TriageClient(RestClient.Builder builder, Validator validator) {
        this.rest = builder.baseUrl("http://ai-service:8000").build();
        this.validator = validator;
    }

    public TicketTriage triage(String message) {
        TicketTriage result = rest.post().uri("/triage")
            .body(Map.of("message", message))
            .retrieve()
            .body(TicketTriage.class);
        var violations = validator.validate(result);
        if (!violations.isEmpty()) throw new IllegalStateException("Invalid triage: " + violations);
        return result;
    }
}
```

Tip: Pydantic uses `snake_case` by default; configure Jackson with `PropertyNamingStrategies.SNAKE_CASE` or use Pydantic aliases to emit `camelCase`.

**Spring AI** offers the same idea natively in Java — for example, mapping a chat response directly to a record via its `.entity(TicketTriage.class)` API — useful when the LLM call lives in the Spring Boot app itself.

---

## 14. Testing and measuring reliability

### 14.1 Unit tests for schemas and validators (no LLM)

```python
# tests/test_receipt_schema.py
import pytest
from pydantic import ValidationError
from schemas.receipt import Receipt

def test_rejects_inconsistent_total():
    with pytest.raises(ValidationError, match="does not match total"):
        Receipt.model_validate({
            "merchant_name": "Lucky Mart", "purchase_date": "2026-01-10", "currency": "USD",
            "items": [{"name": "Milk", "quantity": 2, "unit_price": 1.5, "line_total": 3.0}],
            "total": 9.99,
        })

def test_accepts_valid_receipt():
    r = Receipt.model_validate({
        "merchant_name": "Lucky Mart", "purchase_date": "2026-01-10", "currency": "USD",
        "items": [{"name": "Milk", "quantity": 2, "unit_price": 1.5, "line_total": 3.0}],
        "total": 3.0,
    })
    assert r.total == 3
```

### 14.2 Accuracy evaluation (with the LLM)

Build labeled examples and measure **field-level accuracy**, not just "valid JSON":

```python
# eval/triage_eval.py
import json
from collections import defaultdict
from pathlib import Path
from llm.structured import structured, StructuredOutputError
from schemas.ticket import TicketTriage

cases = [json.loads(l) for l in Path("eval/triage_cases.jsonl").read_text(encoding="utf-8").splitlines()]
fields = ["category", "priority", "language", "needs_human", "order_id"]
correct, failures = defaultdict(int), 0

for case in cases:
    try:
        pred = structured(TicketTriage, user=case["message"],
                          system="You triage customer support messages for KhmerMart.",
                          model="claude-haiku-4-5-20251001")
    except StructuredOutputError:
        failures += 1
        continue
    for f in fields:
        correct[f] += int(getattr(pred, f) == case["expected"][f])

n = len(cases)
print(f"hard failures: {failures}/{n}")
for f in fields:
    print(f"{f:>12}: {correct[f] / n:.1%}")
```

Track these numbers per model and per prompt/schema version. A confusion matrix for `category` and `priority` shows *which* mistakes happen (e.g., "high" vs "urgent" confusion → sharpen the description).

---

## 15. Exercises

1. **Ladder comparison.** Implement ticket triage at Levels 1, 2, 3, and 4. Run each on 100 synthetic messages (generate them with an LLM, including Khmer, typos, and multi-issue messages). Report: parse failure rate, repair rate, field accuracy, latency, tokens.

2. **Schema design A/B.** Create two versions of `TicketTriage`: one with terse field names and no descriptions, one following section 8 (descriptions, reasoning-first, nullable fields). Measure accuracy on the same labeled set.

3. **Evidence verification.** Build a CV extractor (name, email, skills, years of experience per job) where every field includes an `evidence` quote. Write code that flags any field whose evidence isn't found in the source text (after whitespace normalization).

4. **Receipt pipeline.** Photograph or find 15 real receipts (USD and KHR). Extract with the `Receipt` schema and validators. How often do validators catch errors? How often does the repair turn fix them?

5. **Self-querying RAG.** Add `parse_search`-style metadata extraction to your Lesson 5 RAG system (e.g., document category, year, course code). Measure whether recall improves on questions that mention those attributes.

6. **Quiz generator for React.** Design a `Quiz` schema (questions, 4 options, correct index, explanation) with a validator ensuring exactly one correct answer and no duplicate options. Build a FastAPI endpoint and a Next.js page that renders it.

7. **Java boundary.** Expose `/triage` from FastAPI, consume it from Spring Boot with the record above, and write a test that fails when the Python service returns an invalid enum value.

---

## 16. Checklist

- [ ] You can explain why constrained decoding guarantees structure (logit masking) but not correctness
- [ ] You model outputs with Pydantic: enums, nullable fields, descriptions, bounds
- [ ] You can implement all four ladder levels and know when to use each
- [ ] You implemented a validate-and-repair loop that feeds errors back to the model
- [ ] Your schemas put reasoning before decisions and include escape-hatch enum values
- [ ] Business rules live in validators and trigger self-correction or human review
- [ ] You never let an LLM write raw SQL for user queries — it fills a schema, your code builds the query
- [ ] You check `stop_reason` for truncation and refusals
- [ ] You measure field-level accuracy on a labeled set, not just JSON validity
- [ ] Spring Boot validates AI service responses at its own boundary

**Next: Lesson 7 — Tool calling.** Structured output lets the model *describe* an action. Tool calling lets it *take* actions — we'll connect an LLM to real Spring Boot APIs, safely.
