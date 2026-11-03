
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading Qwen model onto GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

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

print("\nLoRA setup successful!")

print("\nTrainable parameter summary:")
model.print_trainable_parameters()

print("\nGPU memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")

print("\nLoRA integration test completed!")

del model
torch.cuda.empty_cache()
