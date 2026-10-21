
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.jsonl"
BATCH_FILE = BASE_DIR / "generated_batch_2.jsonl"
BACKUP_FILE = BASE_DIR / "train_before_batch_2.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def user_question(example):
    for message in example["messages"]:
        if message["role"] == "user":
            return message["content"].strip().casefold()
    raise ValueError("Example has no user message")


def main():
    existing = load_jsonl(TRAIN_FILE)
    batch = load_jsonl(BATCH_FILE)

    seen = {user_question(item) for item in existing}
    new_examples = []

    for item in batch:
        question = user_question(item)

        if question not in seen:
            new_examples.append(item)
            seen.add(question)

    # Preserve the original training file before merging.
    if not BACKUP_FILE.exists():
        BACKUP_FILE.write_text(
            TRAIN_FILE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    with TRAIN_FILE.open("a", encoding="utf-8") as file:
        for item in new_examples:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print("Original examples:", len(existing))
    print("Batch examples:", len(batch))
    print("New examples added:", len(new_examples))
    print("Duplicates skipped:", len(batch) - len(new_examples))
    print("Total examples:", len(existing) + len(new_examples))
    print("Backup:", BACKUP_FILE)


if __name__ == "__main__":
    main()
