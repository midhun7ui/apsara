
import json
import re
from pathlib import Path

path = Path("prepared_data/filtered_experiment/train.jsonl")

placeholder = re.compile(r"\{\{[^{}]+\}\}")
total = 0
invalid = 0
remaining_placeholders = 0
empty_messages = 0

with path.open(encoding="utf-8") as file:
    for line in file:
        try:
            row = json.loads(line)
            messages = row["messages"]

            if (
                len(messages) != 2
                or messages[0]["role"] != "user"
                or messages[1]["role"] != "assistant"
            ):
                invalid += 1
                continue

            user = messages[0]["content"].strip()
            assistant = messages[1]["content"].strip()

            if not user or not assistant:
                empty_messages += 1

            if placeholder.search(user + "\n" + assistant):
                remaining_placeholders += 1

            total += 1

        except (json.JSONDecodeError, KeyError, TypeError):
            invalid += 1

print("Total valid rows:", total)
print("Invalid rows:", invalid)
print("Empty messages:", empty_messages)
print("Rows with unresolved placeholders:", remaining_placeholders)
