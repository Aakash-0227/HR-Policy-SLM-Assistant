
import json
import time
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
OUTPUT_FILE = (
    PROJECT_ROOT / "experiments" / "test_set_baseline_results.json"
)

# Models selected for comparison
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct"
]

SYSTEM_PROMPT = (
    "You are an HR Policy Assistant. "
    "Answer employee questions clearly and concisely. "
    "Do not invent company policies, leave balances, or procedures. "
    "If specific policy information is unavailable, "
    "ask the employee to contact HR or check the official policy."
)

MAX_NEW_TOKENS = 120


def load_test_data():
    """Load the held-out test questions."""

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("Test dataset is empty.")

    return records


def evaluate_model(model_name, test_records):
    """Generate answers and measure performance for one model."""

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available()
            else torch.float32
        ),
        device_map="auto",
        low_cpu_mem_usage=True
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_results = []

    for record in test_records:

        question = record["question"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # Use the model's chat template when available
        if tokenizer.chat_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
        else:
            prompt = (
                SYSTEM_PROMPT
                + "\n\nQuestion: "
                + question
                + "\nAnswer:"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - start_time

        # Decode only newly generated tokens
        generated_tokens = outputs[0][input_length:]

        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        if torch.cuda.is_available():
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        else:
            peak_memory = None

        result = {
            "id": record["id"],
            "category": record["category"],
            "question": question,
            "reference_answer": record["answer"],
            "generated_answer": answer,
            "inference_time_seconds": round(inference_time, 2),
            "peak_gpu_memory_mb": (
                round(peak_memory, 2)
                if peak_memory is not None else None
            )
        }

        model_results.append(result)

        print("\nQuestion:", question)
        print("Answer:", answer)
        print("Inference Time:", round(inference_time, 2), "seconds")

        if peak_memory is not None:
            print("Peak GPU Memory:", round(peak_memory, 2), "MB")

    # Release this model before loading the next one
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_results


def main():

    test_records = load_test_data()

    print("HR POLICY SLM - TEST SET BASELINE EVALUATION")
    print("Total test records:", len(test_records))
    print("Models to evaluate:", len(MODELS))
    print("GPU:", (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ))

    all_results = []

    for model_name in MODELS:

        try:
            model_results = evaluate_model(
                model_name,
                test_records
            )

            all_results.append({
                "model": model_name,
                "status": "completed",
                "results": model_results
            })

        except Exception as error:
            print("\nERROR evaluating", model_name)
            print(type(error).__name__, ":", error)

            all_results.append({
                "model": model_name,
                "status": "failed",
                "error": str(error)
            })

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("TEST SET EVALUATION COMPLETED")
    print("=" * 65)

    for result in all_results:
        print(result["model"], "-", result["status"])

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
