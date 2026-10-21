
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models" / "qwen" / "hf"
EVAL_FILE = BASE_DIR / "evaluation.jsonl"
OUTPUT_FILE = BASE_DIR / "base_model_results.jsonl"


def load_examples(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    examples = load_examples(EVAL_FILE)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    results = []

    for index, example in enumerate(examples, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Apsara, a ride-sharing assistant for Kerala. "
                    "Never invent live ride availability, prices, seat counts, "
                    "or booking status. Ask for missing details and require "
                    "confirmation before a booking. Be honest when live data "
                    "or an app operation is unavailable."
                ),
            },
            *example["messages"],
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = (
            inputs["input_ids"]
            if hasattr(inputs, "keys")
            else inputs
        )
        attention_mask = (
            inputs.get("attention_mask")
            if hasattr(inputs, "get")
            else None
        )

        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=120,
                do_sample=False,
            )

        answer = tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "question": example["messages"][0]["content"],
            "expected_answer": example["messages"][1]["content"],
            "model_answer": answer,
        })

        print(f"[{index}/{len(examples)}] {results[-1]['question']}")
        print("Apsara:", answer, "\n")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Evaluation completed:", len(results))
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
