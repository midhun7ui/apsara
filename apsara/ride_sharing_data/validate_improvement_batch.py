import json
from pathlib import Path

path = Path(__file__).resolve().parent / "malayalam_manglish_improvement_batch.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 30, f"Expected 30 examples, found {len(rows)}"
for i, row in enumerate(rows, 1):
    assert isinstance(row.get("messages"), list), f"Row {i}: messages missing"
    assert any(m.get("role") == "user" for m in row["messages"]), f"Row {i}: user missing"
    assert any(m.get("role") == "assistant" for m in row["messages"]), f"Row {i}: assistant missing"
print(f"Valid JSONL rows: {len(rows)}")
print("Language mix: 15 Manglish + 15 Malayalam-script examples")
print("These are a proposed batch; merge only after reviewing the examples.")
