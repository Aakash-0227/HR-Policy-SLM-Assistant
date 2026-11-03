import json
from pathlib import Path

log_file = Path("experiments/lora_training_log.json")

with open(log_file, "r", encoding="utf-8") as file:
    logs = json.load(file)

print("LoRA Training Loss Summary\n")

for entry in logs:
    print(
        f"Epoch {entry['epoch']}: "
        f"Average loss = {entry['average_training_loss']:.4f}"
    )