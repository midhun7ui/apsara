
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "prepared_data" / "train.jsonl"
TEST_FILE = BASE_DIR / "prepared_data" / "test.jsonl"

def inspect_file(file_path):
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {file_path}")

    if not file_path.exists():
        print("File not found!")
        return

    rows = []
    invalid_rows = 0
    empty_rows = 0
    role_errors = 0
    user_lengths = []
    assistant_lengths = []
    samples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue

            rows.append(row)
            messages = row.get("messages", [])

            if len(messages) != 2:
                role_errors += 1
                continue

            if (
                messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                role_errors += 1
                continue

            user_text = messages[0].get("content", "").strip()
            assistant_text = messages[1].get("content", "").strip()

            if not user_text or not assistant_text:
                empty_rows += 1
                continue

            user_lengths.append(len(user_text))
            assistant_lengths.append(len(assistant_text))

            if len(samples) < 5:
                samples.append((user_text, assistant_text))

    print(f"Total valid JSON rows: {len(rows)}")
    print(f"Invalid JSON lines: {invalid_rows}")
    print(f"Unexpected message structure/roles: {role_errors}")
    print(f"Empty user/assistant messages: {empty_rows}")

    if user_lengths:
        print(f"Average user message length: {sum(user_lengths) / len(user_lengths):.1f} characters")
        print(f"Average assistant response length: {sum(assistant_lengths) / len(assistant_lengths):.1f} characters")

    print("\nSample conversations:")
    for index, (user_text, assistant_text) in enumerate(samples, start=1):
        print(f"\n--- Sample {index} ---")
        print("USER:", user_text[:250])
        print("ASSISTANT:", assistant_text[:400])


inspect_file(TRAIN_FILE)
inspect_file(TEST_FILE)
