
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train_formatted.json"

MAX_LENGTH = 256
LEARNING_RATE = 2e-4

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

torch.manual_seed(42)
torch.cuda.reset_peak_memory_stats()

print("GPU:", torch.cuda.get_device_name(0))

# Load one training record
with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)

record = records[0]
messages = record["messages"]

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare prompt and answer
prompt_text = tokenizer.apply_chat_template(
    messages[:-1],
    tokenize=False,
    add_generation_prompt=True
)

answer = messages[-1]["content"]
full_text = prompt_text + answer + tokenizer.eos_token

prompt_ids = tokenizer(
    prompt_text,
    add_special_tokens=False
)["input_ids"]

full_ids = tokenizer(
    full_text,
    add_special_tokens=False,
    truncation=True,
    max_length=MAX_LENGTH
)["input_ids"]

if full_ids[:len(prompt_ids)] != prompt_ids:
    raise ValueError("Prompt token alignment failed.")

if len(full_ids) <= len(prompt_ids):
    raise ValueError("No assistant response tokens available.")

labels = full_ids.copy()

for i in range(len(prompt_ids)):
    labels[i] = -100

input_ids = torch.tensor(
    [full_ids], dtype=torch.long, device="cuda"
)

attention_mask = torch.ones_like(input_ids)

labels = torch.tensor(
    [labels], dtype=torch.long, device="cuda"
)

print("Input tokens:", input_ids.shape[1])
print("Assistant tokens:", sum(x != -100 for x in labels[0].tolist()))

# Load model
print("\nLoading Qwen model onto GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

model.print_trainable_parameters()

# Optimizer for LoRA parameters only
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

# One forward pass
print("\nRunning forward pass...")

model.train()
optimizer.zero_grad(set_to_none=True)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs.loss

print("Forward pass successful!")
print("Training loss:", round(loss.item(), 4))

# One backward pass
print("\nRunning backward pass...")

loss.backward()

print("Backward pass successful!")

# One optimizer update
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("LoRA optimizer step successful!")

print(
    "\nPeak GPU memory:",
    round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
    "MB"
)

# Clean up
del model, optimizer, outputs, loss
torch.cuda.empty_cache()

print("\nONE-BATCH TRAINING TEST PASSED!")
