
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)

question = "How can an employee apply for sick leave?"

messages = [
    {
        "role": "system",
        "content": "You are an HR assistant. Give clear and helpful answers. Do not invent company policies."
    },
    {
        "role": "user",
        "content": question
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

torch.cuda.synchronize()
latency = time.time() - start_time

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

print("\nQuestion:", question)
print("Answer:", response)
print(f"Inference Time: {latency:.2f} seconds")
print(f"Peak GPU Memory: {peak_memory:.2f} MB")
print("GPU:", torch.cuda.get_device_name(0))
