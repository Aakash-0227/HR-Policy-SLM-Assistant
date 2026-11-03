
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "splits"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

SYSTEM_PROMPT = (
    "You are an HR Policy and Employee Support Assistant. "
    "Answer employee questions clearly and professionally. "
    "Do not invent company policies, employee records, leave balances, "
    "or payroll information. If specific company information is unavailable, "
    "ask the employee to contact HR or check the official company policy."
)


def prepare_training_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = SPLIT_DIR / "train.json"
    output_file = OUTPUT_DIR / "train_formatted.json"

    with open(input_file, "r", encoding="utf-8") as file:
        records = json.load(file)

    formatted_records = []

    for record in records:
        formatted_records.append({
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": record["question"]
                },
                {
                    "role": "assistant",
                    "content": record["answer"]
                }
            ]
        })

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            formatted_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Training data preparation completed!")
    print("Original records:", len(records))
    print("Formatted records:", len(formatted_records))
    print("Output file:", output_file)

    if formatted_records:
        print("\nSample formatted record:")
        print(json.dumps(formatted_records[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    prepare_training_data()
