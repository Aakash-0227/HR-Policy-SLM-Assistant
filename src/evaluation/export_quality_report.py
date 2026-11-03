
import json
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. FILE PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    ROOT_DIR
    / "experiments"
    / "baseline_vs_finetuned_results.json"
)

REPORT_FILE = (
    ROOT_DIR
    / "reports"
    / "answer_quality_comparison.md"
)

# ==========================================
# 2. LOAD EVALUATION RESULTS
# ==========================================

with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

baseline_results = results["baseline_results"]
finetuned_results = results["fine_tuned_results"]

if len(baseline_results) != len(finetuned_results):
    raise ValueError("Baseline and fine-tuned result counts differ.")

if not baseline_results:
    raise ValueError("No evaluation results found.")

# ==========================================
# 3. START REPORT
# ==========================================

report = []

report.append("# HR Policy & Employee Support Assistant")
report.append("## Baseline vs Fine-Tuned Answer Quality Evaluation")

report.append("")
report.append(f"**Evaluation date:** {datetime.now().strftime('%Y-%m-%d')}")
report.append(f"**Base model:** {results['model']}")
report.append("**Fine-tuning method:** LoRA")
report.append(f"**Test dataset:** {results['test_dataset']}")
report.append(f"**Total held-out questions:** {len(baseline_results)}")

report.append("")
report.append("## 1. Evaluation Objective")
report.append("")
report.append(
    "This evaluation compares the original Qwen model with the "
    "Qwen model adapted using LoRA. Both models were evaluated "
    "using the same held-out HR test questions."
)

report.append("")
report.append("## 2. Generation Settings")
report.append("")
report.append("| Setting | Value |")
report.append("|---|---|")

for key, value in results["generation_settings"].items():
    report.append(f"| {key} | {value} |")

report.append("")
report.append("## 3. Answer-by-Answer Comparison")
report.append("")

# ==========================================
# 4. ADD ALL ANSWER COMPARISONS
# ==========================================

for index, (base, tuned) in enumerate(
    zip(baseline_results, finetuned_results),
    start=1
):

    if base["id"] != tuned["id"]:
        raise ValueError(
            f"Test record mismatch at comparison {index}."
        )

    report.append(f"### Test Case {index}: {base['id']}")
    report.append("")
    report.append(f"**Category:** {base['category']}")
    report.append("")
    report.append("**Employee question:**")
    report.append("")
    report.append(base["question"])
    report.append("")
    report.append("**Reference answer:**")
    report.append("")
    report.append(base["reference_answer"])
    report.append("")
    report.append("**Original Qwen answer:**")
    report.append("")
    report.append(base["generated_answer"])
    report.append("")
    report.append("**Fine-tuned Qwen + LoRA answer:**")
    report.append("")
    report.append(tuned["generated_answer"])
    report.append("")
    report.append("| Performance metric | Baseline | Fine-tuned |")
    report.append("|---|---:|---:|")
    report.append(
        f"| Latency (seconds) | "
        f"{base['latency_seconds']} | "
        f"{tuned['latency_seconds']} |"
    )
    report.append(
        f"| Peak GPU memory (MB) | "
        f"{base['peak_gpu_memory_mb']} | "
        f"{tuned['peak_gpu_memory_mb']} |"
    )
    report.append("")
    report.append("---")
    report.append("")

# ==========================================
# 5. PERFORMANCE SUMMARY
# ==========================================

summary = results["summary"]

report.append("## 4. Overall Performance Summary")
report.append("")
report.append("| Metric | Baseline | Fine-tuned |")
report.append("|---|---:|---:|")

for metric, label in [
    ("average_latency_seconds", "Average latency (seconds)"),
    ("average_peak_gpu_memory_mb", "Average peak GPU memory (MB)"),
    ("total_test_questions", "Total test questions")
]:
    base_value = summary["baseline"][metric]
    tuned_value = summary["fine_tuned"][metric]

    report.append(
        f"| {label} | {base_value} | {tuned_value} |"
    )

# ==========================================
# 6. QUALITY REVIEW NOTES
# ==========================================

report.append("")
report.append("## 5. Qualitative Review")
report.append("")
report.append(
    "The generated answers should be manually reviewed for:"
)
report.append("")
report.append("- Relevance to the employee question.")
report.append("- Correctness compared with the reference answer.")
report.append("- Completeness of the response.")
report.append("- Unsupported policy claims or hallucinations.")
report.append("- Clear and professional HR communication.")
report.append("")
report.append(
    "**Note:** Lower training loss or response latency alone "
    "does not prove improved answer quality. No automatic "
    "quality score is assigned by this report."
)

report.append("")
report.append("## 6. Evaluation Limitations")
report.append("")
report.append(
    "- The evaluation uses 15 held-out synthetic HR examples."
)
report.append(
    "- The dataset does not represent an actual company's "
    "official HR policy handbook."
)
report.append(
    "- Manual review is required before claiming a quality improvement."
)
report.append(
    "- Results may vary with hardware load and generation conditions."
)

report.append("")
report.append("## 7. Conclusion")
report.append("")
report.append(
    "This report records the baseline and LoRA fine-tuned "
    "model outputs and their measured performance. Final "
    "conclusions about answer quality must be based on "
    "review of the individual test cases."
)

# ==========================================
# 7. SAVE REPORT
# ==========================================

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("Answer quality report generated successfully!")
print("Total comparisons:", len(baseline_results))
print("Report saved to:", REPORT_FILE)
