
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent
APSARA_DIR = BASE_DIR.parent

MODEL_DIR = APSARA_DIR / "models" / "qwen" / "hf"
DATA_FILE = BASE_DIR / "train.jsonl"
OUTPUT_DIR = APSARA_DIR / "models" / "qwen" / "apsara-lora-v1"
MAX_LENGTH = 384


def load_examples():
    examples = []
    with DATA_FILE.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    if not examples:
        raise ValueError("Training dataset is empty.")

    return examples


def prepare_dataset(tokenizer, examples):
    rows = []

    for example in examples:
        messages = example["messages"]

        prompt_messages = [
            message for message in messages
            if message["role"] != "assistant"
        ]

        answer = next(
            message["content"]
            for message in messages
            if message["role"] == "assistant"
        )

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        answer_text = answer + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        # Keep the complete answer; trim the prompt if necessary.
        if len(answer_ids) >= MAX_LENGTH:
            raise ValueError(
                "An answer exceeds MAX_LENGTH. Increase MAX_LENGTH "
                "or review that example."
            )

        prompt_ids = prompt_ids[-(MAX_LENGTH - len(answer_ids)):]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })

    return rows


class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                [self.pad_token_id] * pad_len + item["input_ids"]
            )
            attention_masks.append(
                [0] * pad_len + item["attention_mask"]
            )
            labels.append([-100] * pad_len + item["labels"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Choose a new output directory to avoid overwriting results."
        )

    torch.set_num_threads(min(torch.get_num_threads(), 6))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False

    print("Applying LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    examples = load_examples()
    dataset = prepare_dataset(tokenizer, examples)

    print("Training examples:", len(dataset))
    print("Maximum sequence length:", MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        use_cpu=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    print("Starting CPU LoRA training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.")
    print("Adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
