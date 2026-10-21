
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.jsonl"
NEW_FILE = BASE_DIR / "improvement_review.jsonl"
BACKUP_FILE = BASE_DIR / "train_before_improvements.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def question_key(record):
    return next(
        message["content"].strip().casefold()
        for message in record["messages"]
        if message["role"] == "user"
    )


def main():
    if not TRAIN_FILE.exists() or not NEW_FILE.exists():
        raise FileNotFoundError("Training or improvement file is missing.")

    train = load_jsonl(TRAIN_FILE)
    new = load_jsonl(NEW_FILE)

    # Validate the two-message structure before modifying anything.
    for record in train + new:
        messages = record.get("messages", [])
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not messages[0].get("content", "").strip()
            or not messages[1].get("content", "").strip()
        ):
            raise ValueError("Invalid example found; merge cancelled.")

    if BACKUP_FILE.exists():
        raise FileExistsError(
            f"Backup already exists: {BACKUP_FILE}. "
            "Inspect it before proceeding."
        )

    existing = {question_key(record) for record in train}
    additions = []
    skipped = 0

    for record in new:
        key = question_key(record)
        if key in existing:
            skipped += 1
            continue

        additions.append(record)
        existing.add(key)

    if not additions:
        print("No new examples to add.")
        return

    shutil.copy2(TRAIN_FILE, BACKUP_FILE)

    with TRAIN_FILE.open("a", encoding="utf-8") as file:
        for record in additions:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Original examples:", len(train))
    print("New examples added:", len(additions))
    print("Duplicates skipped:", skipped)
    print("Total examples:", len(train) + len(additions))
    print("Backup:", BACKUP_FILE)


if __name__ == "__main__":
    main()
