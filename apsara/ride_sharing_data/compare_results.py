
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BASE_FILE = DATA_DIR / "base_model_results.jsonl"
LORA_FILE = DATA_DIR / "lora_gpu_test_results.jsonl"
REPORT_FILE = DATA_DIR / "comparison_report.txt"


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


base_rows = load_jsonl(BASE_FILE)
lora_rows = load_jsonl(LORA_FILE)

base_by_question = {row["question"]: row for row in base_rows}
lora_by_question = {row["question"]: row for row in lora_rows}

lines = [
    f"Base model answers: {len(base_rows)}",
    f"LoRA model answers: {len(lora_rows)}",
    "",
]

for number, (question, base) in enumerate(base_by_question.items(), start=1):
    lora = lora_by_question.get(question)

    lines += [
        "=" * 80,
        f"QUESTION {number}: {question}",
        "",
        "BASE MODEL:",
        base.get("model_answer", ""),
        "",
        "LORA MODEL:",
        lora.get("answer", "[No matching question]") if lora else "[No matching question]",
        "",
    ]

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"Comparison report saved to: {REPORT_FILE}")
print(f"Matched questions: {sum(q in lora_by_question for q in base_by_question)} / {len(base_by_question)}")
