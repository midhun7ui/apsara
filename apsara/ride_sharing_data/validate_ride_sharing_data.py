import json
from pathlib import Path

path = Path(__file__).resolve().parent / "train.jsonl"
required_roles = {"user", "assistant"}

total = 0
errors = []

with path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_number}: Invalid JSON: {exc}")
            continue

        messages = example.get("messages")

        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"Line {line_number}: Invalid messages list")
            continue

        roles = [message.get("role") for message in messages]

        if roles[0] != "user" or roles[-1] != "assistant":
            errors.append(f"Line {line_number}: Expected user then assistant")

        for message in messages:
            if message.get("role") not in required_roles:
                errors.append(f"Line {line_number}: Unexpected role")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(f"Line {line_number}: Empty or invalid content")

        total += 1

print("Total examples:", total)
print("Errors:", len(errors))

for error in errors:
    print(error)

if errors:
    raise SystemExit(1)

print("Dataset structure validation passed.")