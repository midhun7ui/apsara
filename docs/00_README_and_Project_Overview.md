# Apsara — Project Overview

## Goal
Apsara is a ride-sharing customer-service assistant built by fine-tuning `Qwen/Qwen2.5-1.5B-Instruct`. The current training focus is English, Malayalam, and Manglish ride-sharing conversations.

The model is intended to understand rider requests and respond helpfully. It must not invent live driver, route, fare, or booking information. Real ride results will eventually come from a backend/API or MCP-connected tools, not from the language model's memory.

## Current architecture

1. **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
2. **Training method:** LoRA fine-tuning through Hugging Face Transformers and PEFT
3. **Training data:** JSONL chat examples
4. **Held-out evaluation data:** separate JSONL file not used for training
5. **Future runtime:** application backend/tools provide real-time ride information

## Project paths

Local project path used in this workflow:

```text
/home/midhun/LLM-MODEL-E-COMMERCE
```

Kaggle working project:

```text
/kaggle/working/apsara
```

Important files:

```text
apsara/ride_sharing_data/train_manglish_english_v2.jsonl
apsara/ride_sharing_data/evaluation.jsonl
apsara/ride_sharing_data/train_lora_kaggle.py
apsara/ride_sharing_data/evaluate_lora_kaggle_manglish_v2.py
```

## Current verified progress

- Training JSONL validation found **137 examples**, no empty message examples, and no invalid JSON lines.
- The held-out evaluation file contains **20 examples**.
- The Kaggle training run completed **105 optimizer steps across 3 epochs**.
- Reported training runtime: about **131.6 seconds**.
- Reported final training loss: **3.224**.
- The adapter was saved at:

```text
/kaggle/working/apsara/apsara/models/qwen/apsara-lora-kaggle-manglish-v2-20261022
```

- The adapter file `adapter_model.safetensors` was about 8.34 MB.
- **Important:** training completing successfully does not prove that Apsara responds well. The new adapter still needs to be evaluated on the held-out examples.

## Main learning principle

Keep these stages separate:

1. Prepare and validate data.
2. Run a tiny smoke test.
3. Train the adapter.
4. Evaluate on unseen prompts.
5. Inspect failures and improve the data.
6. Compare against the base model.
7. Integrate live tools only after the model's boundaries are clear.
