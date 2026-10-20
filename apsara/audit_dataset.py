
import json
import re
from collections import Counter
from pathlib import Path

FILES = [
    Path("prepared_data/train.jsonl"),
    Path("prepared_data/test.jsonl"),
]

PLACEHOLDER_PATTERN = re.compile(
    r"\{\{[^{}]+\}\}|\[[A-Z][A-Z _-]{2,}\]"
)

# Used only to flag examples for review, not automatic deletion.
PROFANITY_PATTERN = re.compile(
    r"\b(fuck(?:ing|ed|er|ers)?|shit|bitch|asshole)\b",
    re.IGNORECASE,
)


def audit_file(path):
    if not path.exists():
        print(f"\nFile not found: {path}")
        return

    questions = Counter()
    answers = Counter()
    placeholder_rows = []
    profanity_rows = []
    long_answers = []
    total = 0

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
                messages = row["messages"]
                user = messages[0]["content"].strip()
                assistant = messages[1]["content"].strip()
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue

            total += 1
            questions[user.casefold()] += 1
            answers[assistant.casefold()] += 1

            if PLACEHOLDER_PATTERN.search(user + "\n" + assistant):
                placeholder_rows.append(line_number)

            if PROFANITY_PATTERN.search(user):
                profanity_rows.append(line_number)

            if len(assistant) > 2000:
                long_answers.append(line_number)

    print(f"\n{'=' * 55}")
    print(f"FILE: {path}")
    print(f"Rows inspected: {total}")
    print(f"Unique normalized questions: {len(questions)}")
    print(f"Questions repeated: {sum(n - 1 for n in questions.values() if n > 1)}")
    print(f"Answers repeated: {sum(n - 1 for n in answers.values() if n > 1)}")
    print(f"Rows with placeholders: {len(placeholder_rows)}")
    print(f"Questions containing flagged profanity: {len(profanity_rows)}")
    print(f"Responses longer than 2,000 characters: {len(long_answers)}")

    print("\nMost repeated questions:")
    for question, count in questions.most_common(5):
        if count > 1:
            print(f"{count}x: {question[:180]}")

    print("\nPlaceholder examples (line numbers):", placeholder_rows[:20])
    print("Profanity examples (line numbers):", profanity_rows[:20])
    print("Long response examples (line numbers):", long_answers[:20])


for file_path in FILES:
    audit_file(file_path)
