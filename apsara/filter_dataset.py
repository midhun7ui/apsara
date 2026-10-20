
import json
import re
from pathlib import Path

SOURCE = Path("prepared_data")
OUTPUT = Path("prepared_data/filtered_experiment")
OUTPUT.mkdir(parents=True, exist_ok=True)

PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}")

def filter_file(filename):
    source_path = SOURCE / filename
    output_path = OUTPUT / filename

    total = 0
    kept = 0
    removed = 0

    with source_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as output:

        for line in source:
            row = json.loads(line)
            total += 1

            messages = row.get("messages", [])
            text = "\n".join(
                message.get("content", "")
                for message in messages
            )

            if PLACEHOLDER.search(text):
                removed += 1
                continue

            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n{filename}")
    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Removed due to placeholders: {removed}")
    print(f"Saved to: {output_path}")

filter_file("train.jsonl")
