# 03 · Fine-tuning (LoRA / PEFT)

> **Goal:** Know *when* to fine-tune and how to do it cheaply.

## Decision Ladder (try in order)
1. **Prompt engineering** → cheapest, fastest.
2. **RAG** → when the model lacks *knowledge*.
3. **Fine-tuning** → when the model needs a *behavior/format/style/domain skill* consistently, or you need a smaller/cheaper model.
4. **Pretraining** → almost never (huge cost).

## Methods
| Method | What trains | Cost |
|---|---|---|
| Full fine-tuning | All weights | Very high GPU memory |
| **LoRA** | Small low-rank adapter matrices (`W + BA`) | Low — industry default |
| **QLoRA** | LoRA on a 4-bit quantized base model | Fits 7–8B models on one consumer GPU |
| DPO | Learns from preferred vs rejected answers | Alignment/style |

## LoRA with Hugging Face PEFT (sketch)
```python
# pip install transformers datasets peft trl accelerate
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="train.jsonl", split="train")
# each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",        # start small; scale up later
    train_dataset=dataset,
    peft_config=peft_config,
    args=SFTConfig(output_dir="out", num_train_epochs=3, per_device_train_batch_size=4,
                   learning_rate=2e-4, logging_steps=10),
)
trainer.train()
trainer.save_model("out/lora-adapter")
```
> Library APIs (TRL/PEFT) evolve quickly — check their docs for the current argument names.

## Data > Everything
- 500–5,000 **high-quality**, diverse examples usually beat 100k noisy ones.
- Match the exact format you'll use at inference.
- Hold out an eval set; compare against the base model + a good prompt.

## Also Know
- **Quantization** (GGUF, AWQ, 4/8-bit) for cheap inference.
- **Distillation**: big model teaches a small one.
- **Serving**: vLLM, TGI, Ollama, llama.cpp.

## Exercises
1. QLoRA fine-tune a small model on 1,000 examples to answer in a fixed JSON format; compare to prompting.
2. Quantize the result and run it locally with Ollama or llama.cpp.

---
Next → [AI Agents](04-agents.md)
