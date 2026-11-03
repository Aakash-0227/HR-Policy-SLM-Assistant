
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT / "data" / "raw" / "hr_faq_dataset_expanded.json"
)

VALID_CATEGORIES = {
    "leave_management",
    "attendance",
    "payroll",
    "onboarding",
    "company_policies"
}

REQUIRED_FIELDS = {"id", "category", "question", "answer"}


def validate_dataset():

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    errors = []
    ids = []
    questions = []
    categories = Counter()

    for index, record in enumerate(records, start=1):

        if not isinstance(record, dict):
            errors.append(f"Record {index}: Not a JSON object")
            continue

        missing = REQUIRED_FIELDS - record.keys()

        if missing:
            errors.append(
                f"Record {index}: Missing fields {missing}"
            )
            continue

        for field in REQUIRED_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                errors.append(
                    f"Record {index}: Empty or invalid {field}"
                )

        ids.append(record["id"])
        questions.append(record["question"].strip().lower())

        if record["category"] not in VALID_CATEGORIES:
            errors.append(
                f"Record {index}: Invalid category"
            )

        categories[record["category"]] += 1

    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs found")

    if len(questions) != len(set(questions)):
        errors.append("Duplicate questions found")

    print("=" * 55)
    print("EXPANDED HR DATASET VALIDATION")
    print("=" * 55)

    print("Total records:", len(records))
    print("Unique IDs:", len(set(ids)))
    print("Unique questions:", len(set(questions)))

    print("\nCategory Distribution:")

    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")

    print("\nValidation Errors:", len(errors))

    for error in errors:
        print("-", error)

    if not errors:
        print("\nDATASET VALIDATION PASSED")
    else:
        print("\nDATASET VALIDATION FAILED")


if __name__ == "__main__":
    validate_dataset()
