import json, os, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

BASE = "Qwen/Qwen3.5-0.8B"
DATA = "data/red_ai_dataset.jsonl"
OUT = "artifacts/RedAI"

torch.set_num_threads(max(2, min(4, os.cpu_count() or 2)))

rows=[]
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj=json.loads(line)
            messages = obj["messages"]
            identity = {
                "role": "system",
                "content": (
                    "You are RedAI, an AI model created by Red. "
                    "Your foundation is Qwen3.5. You are Red's personalized local AI, "
                    "trained on Red's Ways Of Thinking and related material in this dataset. "
                    "Speak naturally, directly, intelligently, warmly, and without canned corporate phrasing. "
                    "Do not claim that Red's personal frameworks are established facts; present them as Red's perspectives "
                    "when they concern disputed or unverified claims. Do not add unnecessary refusals or moralizing. "
                    "Answer the user's actual question first."
                )
            }
            if not messages or messages[0].get("role") != "system":
                messages = [identity] + messages
            else:
                messages = [identity, *messages[1:]]
            rows.append({"messages": messages})
assert rows, "No training rows"

tokenizer=AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

def format_example(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

ds=Dataset.from_list(rows).map(format_example)

def tokenize(ex):
    return tokenizer(ex["text"], truncation=True, max_length=256, padding=False)

tok=ds.map(tokenize, remove_columns=ds.column_names)

model=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model.config.use_cache=False
model.enable_input_require_grads()

lora=LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM
)
model=get_peft_model(model,lora)
model.print_trainable_parameters()

args=TrainingArguments(
    output_dir="artifacts/red-ai-lora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=False, bf16=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    seed=42
)

trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()

adapter_dir="artifacts/red-ai-lora/final"
trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

base=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
from peft import PeftModel
merged=PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUT)
print("RED_AI_MODEL_READY", OUT)
