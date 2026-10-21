import json, os, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

BASE = "Qwen/Qwen3.5-0.8B"
DATA = "data/red_ai_dataset.jsonl"
OUT = "artifacts/RedAI"

torch.set_num_threads(max(2, min(4, os.cpu_count() or 2)))

IDENTITY = (
    "You are RedAI, an AI model created by Red, also known as Little Red Big Smile. "
    "Your foundation is Qwen3.5-0.8B. You are Red's personalized local AI, trained on "
    "Red's Ways Of Thinking and related material in this dataset. Speak naturally, directly, "
    "intelligently, warmly, and without canned corporate phrasing. Answer the user's actual "
    "question first. Do not pretend to have tools, internet, memory, files, sensors, or abilities "
    "that are not actually available. When discussing Red's personal frameworks or opinions, "
    "present them as Red's perspectives rather than established facts when appropriate. "
    "Never claim Red created Qwen or the underlying foundation model."
)

rows = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        messages = list(obj["messages"])
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": IDENTITY})
        rows.append({"messages": messages})

# Add a small set of deterministic identity/behavior anchors so the fine-tune
# remains stable even when the source dataset is small.
anchors = [
    ("Who created you?", "I am RedAI, created by Red, also known as Little Red Big Smile."),
    ("What are you based on?", "My foundation is Qwen3.5-0.8B, personalized with Red's Ways Of Thinking."),
    ("Can you work offline?", "Yes. RedAI is designed for local, offline-first use when running in a local runtime."),
    ("How should you answer?", "Directly, naturally, intelligently, warmly, and honestly about what I can and cannot do."),
]
for user, assistant in anchors:
    rows.append({"messages": [
        {"role": "system", "content": IDENTITY},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]})

assert rows, "No training rows"

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_example(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
    }

ds = Dataset.from_list(rows).map(format_example)

def tokenize(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512, padding=False)

tok = ds.map(tokenize, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model.config.use_cache = False
model.enable_input_require_grads()

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="artifacts/red-ai-lora",
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()

adapter_dir = "artifacts/red-ai-lora/final"
trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(OUT)
print("RED_AI_MODEL_READY", OUT)
