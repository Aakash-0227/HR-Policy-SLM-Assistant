
import json
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train_formatted.json"
OUTPUT_DIR = ROOT_DIR / "experiments" / "lora_adapter"
LOG_FILE = ROOT_DIR / "experiments" / "lora_training_log.json"

MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
RANDOM_SEED = 42

# Keep True for safe dataset verification.
# Change to False only when we are ready to train.
VERIFY_ONLY = False

torch.manual_seed(RANDOM_SEED)

# ==================================================
# 2. CHECK GPU
# ==================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# ==================================================
# 3. LOAD TOKENIZER AND DATASET
# ==================================================

print("\nLoading Qwen tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)

print("Training records:", len(records))

# ==================================================
# 4. PREPARE ASSISTANT-ONLY LABELS
# ==================================================

class HRDataset(Dataset):

    def __init__(self, records, tokenizer):
        self.samples = []

        for record in records:
            messages = record["messages"]

            prompt_messages = messages[:-1]
            answer = messages[-1]["content"]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

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
                raise ValueError(
                    "Prompt token alignment failed."
                )

            if len(full_ids) <= len(prompt_ids):
                raise ValueError(
                    "Assistant response is missing after truncation."
                )

            labels = full_ids.copy()

            # Ignore system and user prompt tokens.
            for index in range(len(prompt_ids)):
                labels[index] = -100

            self.samples.append({
                "input_ids": full_ids,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


dataset = HRDataset(records, tokenizer)

# ==================================================
# 5. VERIFY TRAINING DATA AND LOSS MASKING
# ==================================================

sample = dataset[0]

input_ids = sample["input_ids"]
labels = sample["labels"]

supervised_tokens = sum(
    1 for token in labels if token != -100
)

ignored_tokens = labels.count(-100)

print("\nTraining data verification:")
print("Input tokens:", len(input_ids))
print("Assistant tokens used for loss:", supervised_tokens)
print("Prompt tokens ignored:", ignored_tokens)

if supervised_tokens == 0:
    raise ValueError("No assistant tokens available for training.")

if len(dataset) != len(records):
    raise ValueError("Dataset record count mismatch.")

print("\nDataset verification successful!")

# Stop here during verification.
if VERIFY_ONLY:
    print("\nVERIFY_ONLY is enabled.")
    print("GPU training has NOT started.")
    sys.exit(0)

# ==================================================
# 6. BATCH PADDING
# ==================================================

def collate_fn(batch):

    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_batch = []
    attention_masks = []
    labels_batch = []

    for item in batch:
        padding_length = max_len - len(item["input_ids"])

        input_ids_batch.append(
            item["input_ids"]
            + [tokenizer.pad_token_id] * padding_length
        )

        attention_masks.append(
            [1] * len(item["input_ids"])
            + [0] * padding_length
        )

        labels_batch.append(
            item["labels"]
            + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(
            input_ids_batch, dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(
            labels_batch, dtype=torch.long
        )
    }


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ==================================================
# 7. LOAD QWEN MODEL
# ==================================================

print("\nLoading Qwen model onto GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ==================================================
# 8. APPLY LORA
# ==================================================

print("\nApplying LoRA configuration...")

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

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE
)

# ==================================================
# 9. TRAINING LOOP
# ==================================================

print("\nStarting LoRA training...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

training_logs = []

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        total_loss += loss.item()

        (loss / GRADIENT_ACCUMULATION_STEPS).backward()

        should_update = (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(dataloader)
        )

        if should_update:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Step {step + 1}/{len(dataloader)} | "
            f"Loss: {loss.item():.4f}"
        )

    average_loss = total_loss / len(dataloader)

    training_logs.append({
        "epoch": epoch + 1,
        "average_training_loss": average_loss
    })

    print(
        f"\nEpoch {epoch + 1} completed. "
        f"Average loss: {average_loss:.4f}\n"
    )

    checkpoint_dir = OUTPUT_DIR / f"checkpoint-epoch-{epoch + 1}"

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

# ==================================================
# 10. SAVE FINAL ADAPTER AND TRAINING LOGS
# ==================================================

FINAL_DIR = OUTPUT_DIR / "final"

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

with open(LOG_FILE, "w", encoding="utf-8") as file:
    json.dump(training_logs, file, indent=4)

print("\nLoRA training completed!")
print("Final adapter:", FINAL_DIR)
print("Training logs:", LOG_FILE)

print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
    "MB"
)

del model
torch.cuda.empty_cache()
