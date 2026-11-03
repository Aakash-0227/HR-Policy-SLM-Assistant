import json
from pathlib import Path

# Dataset file location
DATASET_PATH = Path("data/raw/hr_faq_dataset.jsonl")

REQUIRED_FIELDS = ["id", "category", "question", "answer"]

VALID_CATEGORIES = {
    "leave_management",
    "attendance",
    "payroll",
    "onboarding",
    "company_policies"
}


def validate_dataset():

    records = []
    errors = []

    seen_ids = set()
    seen_questions = set()

    print("=" * 60)
    print("HR DATASET VALIDATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found:", DATASET_PATH)
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                errors.append(
                    f"Line {line_number}: Empty line found."
                )
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                errors.append(
                    f"Line {line_number}: Invalid JSON - {error}"
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    f"Line {line_number}: Record must be a JSON object."
                )
                continue

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in record:
                    errors.append(
                        f"Line {line_number}: Missing field '{field}'."
                    )

            # Check empty values
            for field in REQUIRED_FIELDS:
                if field in record:
                    value = record[field]

                    if not isinstance(value, str) or not value.strip():
                        errors.append(
                            f"Line {line_number}: Invalid or empty '{field}'."
                        )

            # Check duplicate IDs
            record_id = record.get("id")

            if isinstance(record_id, str) and record_id.strip():
                if record_id in seen_ids:
                    errors.append(
                        f"Line {line_number}: Duplicate ID '{record_id}'."
                    )
                else:
                    seen_ids.add(record_id)

            # Check duplicate questions
            question = record.get("question")

            if isinstance(question, str) and question.strip():
                normalized_question = question.strip().lower()

                if normalized_question in seen_questions:
                    errors.append(
                        f"Line {line_number}: Duplicate question found."
                    )
                else:
                    seen_questions.add(normalized_question)

            # Check category
            category = record.get("category")

            if isinstance(category, str) and category not in VALID_CATEGORIES:
                errors.append(
                    f"Line {line_number}: Invalid category '{category}'."
                )

            records.append(record)

    # Validation summary
    print("\nTotal records:", len(records))
    print("Unique IDs:", len(seen_ids))
    print("Unique questions:", len(seen_questions))

    print("\nCategory Distribution:")

    category_counts = {}

    for record in records:
        category = record.get("category")

        if isinstance(category, str):
            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("\nValidation Errors:", len(errors))

    if errors:
        for error in errors:
            print("-", error)

        print("\nDataset validation FAILED.")

    else:
        print("\nDataset validation PASSED.")
        print("All records passed the implemented checks.")

    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()