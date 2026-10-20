
import json
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = (
    "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
)

OUTPUT_DIR = Path("prepared_data")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

# Remove empty questions and responses
dataset = dataset.filter(
    lambda row: bool(row["instruction"].strip())
    and bool(row["response"].strip())
)

# Split data: 95% training, 5% testing
splits = dataset.train_test_split(
    test_size=0.05,
    seed=42,
)

def save_jsonl(data, filename):
    path = OUTPUT_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        for row in data:
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": row["instruction"].strip(),
                    },
                    {
                        "role": "assistant",
                        "content": row["response"].strip(),
                    },
                ]
            }

            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(data)} examples to {path}")

save_jsonl(splits["train"], "train.jsonl")
save_jsonl(splits["test"], "test.jsonl")

print("\nDataset preparation complete!")
print("Training examples:", len(splits["train"]))
print("Testing examples:", len(splits["test"]))
