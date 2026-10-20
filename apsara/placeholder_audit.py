
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FILES = [
    Path("prepared_data/train.jsonl"),
    Path("prepared_data/test.jsonl"),
]

PATTERN = re.compile(r"\{\{([^{}]+)\}\}")

for path in FILES:
    counts = Counter()
    examples = defaultdict(list)

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            messages = row["messages"]

            for message in messages:
                for match in PATTERN.finditer(message["content"]):
                    name = match.group(1).strip()
                    counts[name] += 1

                    if len(examples[name]) < 2:
                        examples[name].append(
                            (line_number, message["role"],
                             message["content"][:300])
                        )

    print(f"\n{'=' * 60}")
    print(f"FILE: {path}")
    print(f"Unique placeholder types: {len(counts)}")
    print(f"Total placeholder occurrences: {sum(counts.values())}")

    for name, count in counts.most_common():
        print(f"\n{{{{{name}}}}}: {count} occurrences")

        for line_number, role, content in examples[name]:
            print(f"  Line {line_number} | {role}: {content}")
