
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="cpu"
)

print("\nChecking attention layer names:\n")

target_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

for name, module in model.named_modules():
    if name.split(".")[-1] in target_names:
        print(name)

print("\nModel layer inspection completed!")
del model
