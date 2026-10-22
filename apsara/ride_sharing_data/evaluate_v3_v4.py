
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path("/kaggle/working/apsara/apsara")
DATA_DIR = PROJECT / "ride_sharing_data"
MODEL_DIR = PROJECT / "models/qwen/hf"

ADAPTERS = {
    "v3": PROJECT / "models/qwen/apsara-lora-gpu-v3-20261023",
    "v4": PROJECT / "models/qwen/apsara-lora-gpu-v4-20261023",
}

EVALUATION_FILE = DATA_DIR / "evaluation.jsonl"
OUTPUT_V3 = DATA_DIR / "apsara_v3_comparison_results.jsonl"
OUTPUT_V4 = DATA_DIR / "apsara_v4_evaluation_results.jsonl"

SYSTEM_PROMPT = (
    "You are Apsara, a helpful ride-sharing assistant. "
    "Ask only for missing pickup, destination, date, and time. "
    "Never invent live ride availability, booking status, prices, "
    "or completed actions. Match the user's language when possible."
)

MAX_NEW_TOKENS = 160


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_adapter(adapter_name, adapter_path, tokenizer):
    print(f"\nLoading base model for {adapter_name}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()

    output_path = OUTPUT_V3 if adapter_name == "v3" else OUTPUT_V4
    records = []

    for index, row in enumerate(load_jsonl(EVALUATION_FILE), start=1):
        messages = row["messages"]

        # Use user messages only; never include reference assistant answers.
        user_messages = [
            {"role": "user", "content": m["content"]}
            for m in messages
            if m.get("role") == "user"
        ]

        if not user_messages:
            print(f"Skipping example {index}: no user message")
            continue

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *user_messages,
        ]

        encoded = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0, inputs.shape[1]:]
        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        records.append({
            "index": index,
            "messages": messages,
            "response": response,
            "adapter": adapter_name,
            "system_prompt": SYSTEM_PROMPT,
            "generation": {
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        })

        print(f"\n[{adapter_name}] Example {index}")
        print("User:", user_messages[-1]["content"])
        print("Assistant:", response)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} results to {output_path}")

    del model
    del base_model
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    for path in [
        MODEL_DIR,
        EVALUATION_FILE,
        *ADAPTERS.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    # Do not overwrite existing comparison results.
    if OUTPUT_V3.exists() or OUTPUT_V4.exists():
        raise FileExistsError(
            "A comparison output already exists. Rename or remove the "
            "old comparison files before rerunning."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for adapter_name, adapter_path in ADAPTERS.items():
        evaluate_adapter(adapter_name, adapter_path, tokenizer)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
