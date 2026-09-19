# 04 · AI Agents & Tool Use

> **Goal:** Build LLM systems that take actions: call APIs, query databases, run multi-step tasks.

## What's an Agent?
An LLM in a loop: **think → choose a tool → observe the result → repeat → answer**.

```
User goal → LLM ──(tool call)──▶ Your code runs tool ──(result)──▶ LLM ... ▶ Final answer
```

## Tool Use (function calling)
```python
import anthropic, json
client = anthropic.Anthropic()

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 33, "condition": "sunny"}   # call a real API here

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]

messages = [{"role": "user", "content": "Should I bring an umbrella in Phnom Penh today?"}]
while True:
    resp = client.messages.create(model="claude-sonnet-5", max_tokens=1000,
                                  tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason != "tool_use":
        print(resp.content[0].text)
        break
    results = []
    for block in resp.content:
        if block.type == "tool_use":
            output = get_weather(**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(output)})
    messages.append({"role": "user", "content": results})
```

## Design Patterns
| Pattern | Use |
|---|---|
| Single agent + tools | Most tasks — start here |
| Workflow / chain | Fixed steps (more reliable than a free agent) |
| Router | Classify request → send to specialized prompt/agent |
| Orchestrator + sub-agents | Big tasks split into parallel subtasks |
| Evaluator–optimizer | One LLM drafts, another critiques |

**Rule:** use the simplest pattern that works. Deterministic workflows beat autonomous agents when steps are known.

## Production Concerns
- **Guardrails**: validate tool inputs, allow-list actions, human approval for destructive ones.
- **Limits**: max iterations, timeouts, cost budgets.
- **Observability**: log every step (Langfuse, LangSmith, OpenTelemetry).
- **Evals**: task success rate on a fixed scenario set.
- **Protocols/frameworks**: MCP (standard way to expose tools), LangGraph, provider agent SDKs.
- **Security**: prompt injection from tool outputs/web pages — treat retrieved content as data, not instructions.

## Exercises
1. Build an agent with 3 tools: SQL query (read-only), web search, calculator.
2. Add a max-steps limit, logging, and a 20-task eval suite.

---
Next → [Reinforcement Learning](05-reinforcement-learning.md)
