import json
import time
import gc
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT_DIR / "data" / "splits" / "test.json"
ADAPTER_DIR = ROOT_DIR / "experiments" / "lora_adapter" / "final"
OUTPUT_FILE = ROOT_DIR / "experiments" / "generation_optimization_results.json"

MAX_INPUT_LENGTH = 512
STANDARD_MAX_NEW_TOKENS = 120
OPTIMIZED_MAX_NEW_TOKENS = 80
WARMUP_RUNS = 2

SYSTEM_PROMPT = (
    "You are an HR Policy and Employee Support Assistant. "
    "Answer employee questions clearly and professionally. "
    "Do not invent company policies, employee records, leave balances, "
    "or payroll information. If specific company information is unavailable, "
    "ask the employee to contact HR or check the official company policy."
)

def sync():
    torch.cuda.synchronize()

def generate(model, tokenizer, question, device, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    sync()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    sync()
    latency = time.perf_counter() - start
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return {
        "generated_answer": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "generated_tokens": int(new_ids.shape[0]),
        "latency_seconds": round(latency, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
    }

def summary(items):
    return {
        "average_latency_seconds": round(sum(x["latency_seconds"] for x in items) / len(items), 3),
        "average_peak_gpu_memory_mb": round(sum(x["peak_gpu_memory_mb"] for x in items) / len(items), 2),
        "average_generated_tokens": round(sum(x["generated_tokens"] for x in items) / len(items), 2),
        "total_test_questions": len(items),
    }

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) != 15:
        raise ValueError(f"Expected 15 held-out records; found {len(records)}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading fine-tuned model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("Warm-up runs...")
    for i in range(2):
        q = records[i]["question"]
        generate(model, tokenizer, q, device, STANDARD_MAX_NEW_TOKENS)
        generate(model, tokenizer, q, device, OPTIMIZED_MAX_NEW_TOKENS)

    standard_results, optimized_results = [], []
    print("Running 15 held-out questions...")
    for i, record in enumerate(records, start=1):
        common = {
            "id": record["id"],
            "category": record["category"],
            "question": record["question"],
            "reference_answer": record["answer"],
        }
        # Alternate order to reduce measurement-order bias.
        if i % 2:
            standard = generate(model, tokenizer, record["question"], device, STANDARD_MAX_NEW_TOKENS)
            optimized = generate(model, tokenizer, record["question"], device, OPTIMIZED_MAX_NEW_TOKENS)
        else:
            optimized = generate(model, tokenizer, record["question"], device, OPTIMIZED_MAX_NEW_TOKENS)
            standard = generate(model, tokenizer, record["question"], device, STANDARD_MAX_NEW_TOKENS)

        standard_results.append({**common, **standard})
        optimized_results.append({**common, **optimized})
        print(f"{i}/15 | standard {standard['latency_seconds']}s | optimized {optimized['latency_seconds']}s")

    standard_summary = summary(standard_results)
    optimized_summary = summary(optimized_results)
    reduction = (
        (standard_summary["average_latency_seconds"] - optimized_summary["average_latency_seconds"])
        / standard_summary["average_latency_seconds"] * 100
        if standard_summary["average_latency_seconds"] else 0
    )

    output = {
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_DIR.relative_to(ROOT_DIR)),
        "test_dataset": "data/splits/test.json",
        "method": "Same fine-tuned model and held-out questions; two warm-up runs per setting; measurement order alternated.",
        "standard_configuration": {
            "max_new_tokens": STANDARD_MAX_NEW_TOKENS,
            "do_sample": False, "use_cache": True,
            "summary": standard_summary, "results": standard_results
        },
        "optimized_configuration": {
            "max_new_tokens": OPTIMIZED_MAX_NEW_TOKENS,
            "do_sample": False, "use_cache": True,
            "summary": optimized_summary, "results": optimized_results
        },
        "latency_reduction_percent": round(reduction, 2),
        "quality_review_required": True
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\n===== M5 OPTIMIZATION SUMMARY =====")
    print("Standard:", json.dumps(standard_summary, indent=2))
    print("Optimized:", json.dumps(optimized_summary, indent=2))
    print("Latency reduction (%):", round(reduction, 2))
    print("Saved:", OUTPUT_FILE)
    print("Review answer quality: an 80-token limit may truncate some answers.")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
