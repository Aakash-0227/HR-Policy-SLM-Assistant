
import torch
import time
import gc
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct"
]

QUESTION = "How can an employee apply for sick leave?"

SYSTEM_PROMPT = (
    "You are an HR assistant. Give clear and helpful answers. "
    "Do not invent company policies. If policy information is unavailable, "
    "explain that the employee should check the official HR policy or contact HR."
)

results = []

for model_name in MODELS:

    print("\n" + "=" * 60)
    print("Testing:", model_name)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    torch.cuda.synchronize()
    latency = time.time() - start_time

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )

    result = {
        "model": model_name,
        "question": QUESTION,
        "answer": answer,
        "inference_time_seconds": round(latency, 2),
        "peak_gpu_memory_mb": round(peak_memory, 2)
    }

    results.append(result)

    print("\nAnswer:", answer)
    print("Inference Time:", round(latency, 2), "seconds")
    print("Peak GPU Memory:", round(peak_memory, 2), "MB")

    del model, tokenizer, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

with open(
    "experiments/standardized_benchmark_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print("\nAll model benchmarks completed.")
print("Results saved in experiments/standardized_benchmark_results.json")
