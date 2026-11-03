
import csv
import re
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "reports" / "answer_quality_comparison.md"
OUTPUT_FILE = ROOT_DIR / "reports" / "manual_quality_evaluation.csv"

# --------------------------------------------------
# 2. SCORING CRITERIA
# --------------------------------------------------

CRITERIA = [
    "accuracy",
    "relevance",
    "completeness",
    "unsupported_claims",
    "professional_communication",
]

MAX_SCORE_PER_CRITERION = 2
MAX_TOTAL_SCORE = 10


# --------------------------------------------------
# 3. EXTRACT TEST CASES FROM THE EXISTING REPORT
# --------------------------------------------------

def extract_test_cases(content):
    sections = re.split(
        r"(?=### Test Case \d+:)",
        content
    )

    test_cases = []

    for section in sections:

        heading = re.search(
            r"### Test Case (\d+):\s*(HR\d+)",
            section
        )

        if not heading:
            continue

        def extract(pattern):
            match = re.search(
                pattern,
                section,
                flags=re.IGNORECASE | re.DOTALL
            )

            if match:
                return match.group(1).strip()

            return ""

        category = extract(
            r"\*\*Category:\*\*\s*(.*?)\n"
        )

        question = extract(
            r"\*\*Employee question:\*\*\s*(.*?)\n"
        )

        reference = extract(
            r"\*\*Reference answer:\*\*\s*(.*?)\n\n"
        )

        original = extract(
            r"\*\*Original Qwen answer:\*\*\s*(.*?)"
            r"\n\n\*\*Fine-tuned Qwen \+ LoRA answer:\*\*"
        )

        finetuned = extract(
            r"\*\*Fine-tuned Qwen \+ LoRA answer:\*\*\s*(.*?)"
            r"\n\n\| Performance metric"
        )

        if not question or not reference:
            print(
                f"Warning: Could not extract complete "
                f"content for {heading.group(2)}"
            )
            continue

        test_cases.append({
            "test_number": heading.group(1),
            "test_id": heading.group(2),
            "category": category,
            "question": question,
            "reference_answer": reference,
            "original_answer": original,
            "finetuned_answer": finetuned,
        })

    return test_cases


# --------------------------------------------------
# 4. CREATE MANUAL EVALUATION SCORECARD
# --------------------------------------------------

def create_scorecard(test_cases):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "test_number",
        "test_id",
        "category",
        "question",
        "reference_answer",
        "original_answer",
        "finetuned_answer",
        "original_accuracy",
        "original_relevance",
        "original_completeness",
        "original_unsupported_claims",
        "original_professional_communication",
        "original_total",
        "finetuned_accuracy",
        "finetuned_relevance",
        "finetuned_completeness",
        "finetuned_unsupported_claims",
        "finetuned_professional_communication",
        "finetuned_total",
        "observations",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for case in test_cases:

            row = {
                "test_number": case["test_number"],
                "test_id": case["test_id"],
                "category": case["category"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "original_answer": case["original_answer"],
                "finetuned_answer": case["finetuned_answer"],
                "observations": "",
            }

            for criterion in CRITERIA:
                row[f"original_{criterion}"] = ""
                row[f"finetuned_{criterion}"] = ""

            row["original_total"] = ""
            row["finetuned_total"] = ""

            writer.writerow(row)


# --------------------------------------------------
# 5. PRINT EVALUATION SUMMARY
# --------------------------------------------------

def main():

    print("=" * 65)
    print("HR POLICY & EMPLOYEE SUPPORT ASSISTANT")
    print("MANUAL ANSWER QUALITY EVALUATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        print("ERROR: Evaluation report was not found.")
        print("Expected location:", INPUT_FILE)
        return

    content = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    test_cases = extract_test_cases(content)

    if not test_cases:
        print("ERROR: No test cases could be extracted.")
        print("Check the headings in the evaluation report.")
        return

    create_scorecard(test_cases)

    print("\nEvaluation scorecard created successfully!")

    print("\nTotal test cases:", len(test_cases))

    print("\nTest cases extracted:")

    for case in test_cases:
        print(
            f"{case['test_number']}. "
            f"{case['test_id']} - "
            f"{case['category']}"
        )

    print("\nScoring criteria:")

    for criterion in CRITERIA:
        print(f"- {criterion}: 0 to 2 points")

    print("\nMaximum score per model:", MAX_TOTAL_SCORE)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("Enter manual scores in the generated CSV.")
    print("Do not score based on latency alone.")
    print("Review correctness and unsupported policy claims.")
    print("\nManual evaluation preparation completed!")


if __name__ == "__main__":
    main()
