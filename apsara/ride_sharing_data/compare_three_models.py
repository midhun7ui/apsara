
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BASE_FILE = DATA_DIR / "base_model_results.jsonl"
OLD_LORA_FILE = DATA_DIR / "lora_gpu_test_results.jsonl"
NEW_LORA_FILE = DATA_DIR / "lora_cpu_manglish_v2_results.jsonl"
REPORT_FILE = DATA_DIR / "three_model_comparison_report.txt"


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


base_rows = load_jsonl(BASE_FILE)
old_lora_rows = load_jsonl(OLD_LORA_FILE)
new_lora_rows = load_jsonl(NEW_LORA_FILE)

base_by_question = {row["question"]: row for row in base_rows}
old_by_question = {row["question"]: row for row in old_lora_rows}
new_by_question = {row["question"]: row for row in new_lora_rows}

lines = [
    f"Base model answers: {len(base_rows)}",
    f"Old LoRA answers: {len(old_lora_rows)}",
    f"New Manglish v2 LoRA answers: {len(new_lora_rows)}",
    "",
]

matched = 0

for number, (question, base) in enumerate(base_by_question.items(), start=1):
    old = old_by_question.get(question)
    new = new_by_question.get(question)

    if old and new:
        matched += 1

    lines += [
        "=" * 80,
        f"QUESTION {number}: {question}",
        "",
        "BASE MODEL:",
        base.get("model_answer", ""),
        "",
        "OLD LORA MODEL:",
        old.get("answer", "[No matching question]") if old else "[No matching question]",
        "",
        "NEW MANGLISH V2 LORA MODEL:",
        new.get("answer", "[No matching question]") if new else "[No matching question]",
        "",
    ]

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

print(f"Comparison report saved to: {REPORT_FILE}")
print(f"Questions matched across all three models: {matched} / {len(base_by_question)}")