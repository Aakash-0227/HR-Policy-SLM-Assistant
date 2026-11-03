
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ROOT_DIR = Path(__file__).resolve().parents[2]

TEST_FILE = ROOT_DIR / "data" / "splits" / "test.json"
ADAPTER_DIR = ROOT_DIR / "experiments" / "lora_adapter" / "final"

OUTPUT_DIR = ROOT_DIR / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "baseline_vs_finetuned_results.json"

MAX_NEW_TOKENS = 120
MAX_INPUT_LENGTH = 512

SYSTEM_PROMPT = (
    "You are an HR Policy and Employee Support Assistant. "
    "Answer employee questions clearly and professionally. "
    "Do not invent company policies, employee records, leave balances, "
    "or payroll information. If specific company information is unavailable, "
    "ask the employee to contact HR or check the official company policy."
)

# ==================================================
# 2. CHECK ENVIRONMENT
# ==================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this evaluation.")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# ==================================================
# 3. LOAD HELD-OUT TEST DATA
# ==================================================

with open(TEST_FILE, "r", encoding="utf-8") as file:
    test_records = json.load(file)

if len(test_records) != 15:
    raise ValueError(
        f"Expected 15 held-out records, found {len(test_records)}."
    )

print("\nHeld-out test records:", len(test_records))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 4. LOAD TOKENIZER
# ==================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ==================================================
# 5. PREPARE PROMPT
# ==================================================

def prepare_prompt(question):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


# ==================================================
# 6. GENERATE ANSWER AND MEASURE PERFORMANCE
# ==================================================

def generate_answer(model, question):

    inputs = prepare_prompt(question)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start_time

    # Decode only newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[0][prompt_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    peak_memory = (
        torch.cuda.max_memory_allocated(0) / 1024**2
    )

    return {
        "generated_answer": answer,
        "latency_seconds": round(latency, 3),
        "peak_gpu_memory_mb": round(peak_memory, 2)
    }


# ==================================================
# 7. LOAD BASELINE MODEL
# ==================================================

print("\nLoading original Qwen baseline model...")

baseline_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

baseline_model.eval()

baseline_results = []

print("\nEvaluating baseline model...")

for index, record in enumerate(test_records, start=1):

    result = generate_answer(
        baseline_model,
        record["question"]
    )

    baseline_results.append({
        "id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "reference_answer": record["answer"],
        **result
    })

    print(
        f"Baseline {index}/{len(test_records)} | "
        f"Latency: {result['latency_seconds']} sec"
    )


# ==================================================
# 8. RELEASE BASELINE MODEL
# ==================================================

del baseline_model

torch.cuda.empty_cache()

# ==================================================
# 9. LOAD FINE-TUNED MODEL
# ==================================================

print("\nLoading Qwen with trained LoRA adapter...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

finetuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR
)

finetuned_model.eval()

finetuned_results = []

print("\nEvaluating fine-tuned model...")

for index, record in enumerate(test_records, start=1):

    result = generate_answer(
        finetuned_model,
        record["question"]
    )

    finetuned_results.append({
        "id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "reference_answer": record["answer"],
        **result
    })

    print(
        f"Fine-tuned {index}/{len(test_records)} | "
        f"Latency: {result['latency_seconds']} sec"
    )


# ==================================================
# 10. CALCULATE SUMMARY METRICS
# ==================================================

def calculate_summary(results):

    average_latency = sum(
        item["latency_seconds"] for item in results
    ) / len(results)

    average_memory = sum(
        item["peak_gpu_memory_mb"] for item in results
    ) / len(results)

    return {
        "average_latency_seconds": round(average_latency, 3),
        "average_peak_gpu_memory_mb": round(average_memory, 2),
        "total_test_questions": len(results)
    }


summary = {
    "baseline": calculate_summary(baseline_results),
    "fine_tuned": calculate_summary(finetuned_results)
}

# ==================================================
# 11. SAVE RESULTS
# ==================================================

evaluation_output = {
    "model": MODEL_NAME,
    "adapter": str(ADAPTER_DIR.relative_to(ROOT_DIR)),
    "test_dataset": "data/splits/test.json",
    "generation_settings": {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False
    },
    "baseline_results": baseline_results,
    "fine_tuned_results": finetuned_results,
    "summary": summary
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        evaluation_output,
        file,
        indent=4,
        ensure_ascii=False
    )

# ==================================================
# 12. DISPLAY SUMMARY
# ==================================================

print("\n======================================")
print("BASELINE VS FINE-TUNED MODEL RESULTS")
print("======================================")

print("\nBaseline:")
print(json.dumps(summary["baseline"], indent=4))

print("\nFine-tuned:")
print(json.dumps(summary["fine_tuned"], indent=4))

print("\nEvaluation completed successfully!")
print("Results saved to:", OUTPUT_FILE)

# Cleanup
del finetuned_model, base_model
torch.cuda.empty_cache()
