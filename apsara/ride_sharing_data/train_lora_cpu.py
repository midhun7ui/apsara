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
DATA_FILE = BASE_DIR / "train_manglish_english_v2.jsonl"

# Use a fresh directory for every test run. Override with APSARA_OUTPUT_DIR if needed.
OUTPUT_DIR = Path(
    __import__("os").environ.get(
        "APSARA_OUTPUT_DIR",
        str(APSARA_DIR / "models" / "qwen" / "apsara-lora-cpu-full-20261022"),
    )
)

MAX_LENGTH = 256


def load_examples():
    examples = []
    with DATA_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {DATA_FILE}, line {line_number}: {exc}"
                ) from exc

    if not examples:
        raise ValueError("Training dataset is empty.")

    return examples


def prepare_dataset(tokenizer, examples):
    """Prepare assistant-only labels, trimming prompts and skipping overlong answers."""
    rows = []
    skipped = 0

    for index, example in enumerate(examples, start=1):
        messages = example.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Example {index} has no valid 'messages' list.")

        assistant_messages = [
            message for message in messages
            if message.get("role") == "assistant"
        ]
        if not assistant_messages:
            raise ValueError(f"Example {index} has no assistant message.")

        # Use the final assistant response if a conversation contains more than one.
        answer = assistant_messages[-1].get("content", "")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Example {index} has an empty assistant answer.")

        prompt_messages = [
            message for message in messages
            if message.get("role") != "assistant"
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        answer_text = answer + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        # The answer is never truncated. Skip it if it cannot fit in the sequence.
        if len(answer_ids) >= MAX_LENGTH:
            print(
                f"Skipping example {index}: answer has {len(answer_ids)} tokens "
                f"(MAX_LENGTH={MAX_LENGTH})."
            )
            skipped += 1
            continue

        prompt_budget = MAX_LENGTH - len(answer_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        input_ids = prompt_ids + answer_ids

        # Only the assistant answer contributes to the loss.
        labels = [-100] * len(prompt_ids) + answer_ids

        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })

    print(f"Prepared {len(rows)} examples; skipped {skipped} overlong answers.")
    return rows


class CausalCollator:
    """Left-pad batches and mask padding from the loss."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_masks, labels = [], [], []

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
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    if (OUTPUT_DIR / "adapter_model.safetensors").exists():
        raise FileExistsError(f"Adapter already exists: {OUTPUT_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access the GPU in this environment. "
            "Check the ROCm installation before training."
        )


    torch.set_num_threads(min(torch.get_num_threads(), 6))
    torch.manual_seed(42)

    print("Training device: CPU (FP32)")
    print("Output directory:", OUTPUT_DIR)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model onto CPU in FP32...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cpu")
    model.config.use_cache = False

    print("Applying LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = prepare_dataset(tokenizer, load_examples())
    if not dataset:
        raise ValueError("No training examples fit MAX_LENGTH.")

    print("Training examples:", len(dataset))
    print("Maximum sequence length:", MAX_LENGTH)
    print("Full CPU training: 3 epochs")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        use_cpu=True,
        bf16=False,
        fp16=False,
        num_train_epochs=3,
        max_steps=-1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=42,
        # Prevent Trainer from using DataParallel on multiple visible GPUs.
        # The pre-launch visibility check above is the primary guard.
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    print("Starting full CPU LoRA training...")
    trainer.train()
    print("Training step completed.")


    # Save only after a successful step; never overwrite a previous test.
# Save only after a successful training step.
        # Save the adapter after a successful training step.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Full CPU training adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()



