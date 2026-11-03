
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_FILE = ROOT_DIR / "data" / "raw" / "hr_faq_dataset_expanded.json"
SPLIT_DIR = ROOT_DIR / "data" / "splits"

RANDOM_SEED = 42


def split_dataset():
    # Load expanded dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    categories = [record["category"] for record in records]

    # First, separate 15% for testing
    train_val, test = train_test_split(
        records,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=categories
    )

    # Then separate 15% of the full dataset for validation
    train_val_categories = [record["category"] for record in train_val]

    train, validation = train_test_split(
        train_val,
        test_size=15 / 85,
        random_state=RANDOM_SEED,
        stratify=train_val_categories
    )

    # Create output directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all three splits
    splits = {
        "train.json": train,
        "validation.json": validation,
        "test.json": test
    }

    for filename, data in splits.items():
        output_file = SPLIT_DIR / filename

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        print(f"{filename}: {len(data)} records")

    # Display category distribution
    print("\nCategory distribution:")

    for split_name, data in splits.items():
        counts = Counter(record["category"] for record in data)
        print(f"\n{split_name}:")

        for category, count in sorted(counts.items()):
            print(f"  {category}: {count}")

    print("\nTotal records:", len(train) + len(validation) + len(test))
    print("Dataset splitting completed successfully!")


if __name__ == "__main__":
    split_dataset()
