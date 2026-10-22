import json
from pathlib import Path

data_dir = Path(__file__).resolve().parent

files = [
    "base_model_results.jsonl",
    "lora_gpu_test_results.jsonl",
]

for filename in files:
    path = data_dir / filename
    print(f"\n--- {filename} ---")

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                print(json.dumps(row, ensure_ascii=False, indent=2))
                break
