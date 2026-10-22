
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
APSARA_DIR = BASE_DIR.parent

MODEL_DIR = APSARA_DIR / "models" / "qwen" / "hf"
ADAPTER_DIR = APSARA_DIR / "models" / "qwen" / "apsara-lora-gpu-test-3"
EVAL_FILE = BASE_DIR / "evaluation.jsonl"
OUTPUT_FILE = BASE_DIR / "lora_gpu_test_results.jsonl"

def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE}")

    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Adapter weights not found: {ADAPTER_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available in this environment.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    with EVAL_FILE.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    results = []

    for index, example in enumerate(examples, start=1):
        messages = example["messages"]
        prompt_messages = [
            message for message in messages
            if message["role"] != "assistant"
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda:0")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        results.append({
            "index": index,
            "question": prompt_messages[-1]["content"],
            "answer": answer,
        })

        print(f"\n[{index}/{len(examples)}] {answer}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} results to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
