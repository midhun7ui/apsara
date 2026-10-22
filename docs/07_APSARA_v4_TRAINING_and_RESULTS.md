# 07 — Apsara v4: Training and Evaluation Results

## 1. Overview

Apsara is a multilingual ride-sharing customer-service assistant fine-tuned from `Qwen/Qwen2.5-1.5B-Instruct`.

Version 4 (v4) is an experimental LoRA fine-tuning iteration focused on improving responses to ride-sharing requests in:

- English
- Malayalam
- Manglish (Malayalam written using Latin characters)

The goal is to make Apsara understand ride-related requests, ask for missing journey details, and avoid inventing booking status, availability, prices, or completed actions.

## 2. Base Model

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model family | Qwen2.5 |
| Parameter scale | 1.5B |
| Fine-tuning method | LoRA |
| Training environment | Kaggle |
| GPU | NVIDIA Tesla T4 |
| GPU VRAM | Approximately 14.56 GB |

The base model provides the original pretrained language capabilities. The v4 adapter contains the learned fine-tuning updates.

## 3. Dataset Preparation

The v4 training dataset was prepared from ride-sharing conversation examples in English, Malayalam, and Manglish.

### Dataset versions

| Dataset | Examples | Purpose |
|---|---:|---|
| `train_manglish_english_v2.jsonl` | 137 | Original dataset |
| `train_combined_v1.jsonl` | 167 | Combined dataset |
| `train_combined_v2_candidate.jsonl` | 183 | Candidate dataset with 16 additional cleaned examples |
| `train_combined_v2_deduplicated.jsonl` | 180 | Deduplicated dataset used for v4 |

Three duplicate examples were removed, leaving 180 examples.

### Tokenization validation

- Maximum sequence length: 512 tokens
- Maximum observed token length: 252
- Examples skipped for exceeding the limit: 0

The deduplicated dataset was selected for v4 training.

## 4. LoRA Training Configuration

The v4 training script was:

`ride_sharing_data/train_lora_gpu_v2.py`

The script was configured to use the deduplicated dataset and to save the resulting adapter in the v4 output directory.

| Parameter | Configuration |
|---|---|
| Training epochs | 3 |
| Learning rate | `2e-5` |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | Query, key, value and output projections |
| Maximum sequence length | 512 |
| Loss masking | Assistant-only loss |
| Output directory | `models/qwen/apsara-lora-gpu-v4-20261023` |

The model was trained using the Tesla T4 GPU available in the Kaggle environment.

### Training outcome

The v4 training process completed successfully with exit code 0.

Reported peak GPU memory usage was approximately 3.61 GB.

The adapter weights were saved successfully.

## 5. Output Artifacts

The v4 adapter directory is:

`models/qwen/apsara-lora-gpu-v4-20261023/`

The adapter weights file is:

`adapter_model.safetensors`

The adapter weights file was approximately 8.34 MB.

The tokenizer file in the adapter directory was approximately 10.89 MB.

These files are lightweight compared with the full base model.

**Important:** The adapter is not a complete standalone model. It must be loaded with a compatible base model. The availability of the complete adapter configuration and all required tokenizer files should be verified before deployment.

## 6. Evaluation Methodology

The v3 and v4 adapters were evaluated on the same set of 20 prompts.

The evaluation used the same system prompt:

> You are Apsara, a helpful ride-sharing assistant. Ask only for missing pickup, destination, date, and time. Never invent live ride availability, booking status, prices, or completed actions. Match the user's language when possible.

The evaluation used greedy decoding and a maximum generation length of 160 new tokens.

The comparison scripts and output files included:

- `ride_sharing_data/evaluate_v3_v4.py`
- `apsara_v3_comparison_results.jsonl`
- `apsara_v4_evaluation_results.jsonl`
- `apsara_v3_evaluation_results.jsonl`

The v3 and v4 outputs were observed to be almost identical on the evaluated prompts.

No verified aggregate accuracy score or statistically significant improvement was established by these results.

## 7. Observed Limitations

The evaluation identified several important failure modes.

### 7.1 False booking confirmation

When asked whether a booking was confirmed, the model could respond as though the booking had been confirmed without reliable evidence.

Expected behavior:

Apsara should explain that it cannot verify booking status unless it has access to a trusted booking system or explicit booking data.

### 7.2 Duplicate booking risk

After a timeout, the model could advise the user to retry a booking immediately.

Expected behavior:

Apsara should first check the existing booking status or explain how to verify it before recommending another booking attempt.

### 7.3 Child passenger safety

The model could say that a child may travel alone without checking the applicable service policy or safety requirements.

Expected behavior:

Apsara should avoid making unsupported assurances and should direct the user to the relevant ride-service policy or support channel.

### 7.4 Malayalam and Manglish quality

Some generated Malayalam responses were garbled, and some Manglish responses were unnatural or unclear.

Expected behavior:

Apsara should respond in the user's language where possible, using clear and natural wording.

## 8. Interpretation of v4

Successful training does not guarantee improved real-world behavior.

The current evaluation suggests that v4 remains experimental. Its outputs were very similar to v3, and the observed failure modes remain unresolved.

The small evaluation set of 20 prompts is useful for identifying problems but is not sufficient to establish broad language quality, safety, or production readiness.

The next iteration should focus on measurable improvements rather than simply increasing the number of training epochs.

## 9. Recommended Improvements for v5

### Dataset improvements

Add high-quality examples covering:

- Booking-status verification
- Booking timeouts and duplicate-booking prevention
- Ride cancellation and modification
- Missing pickup and destination information
- Unsupported price and availability questions
- Child passenger and service-policy questions
- Natural Malayalam responses
- Natural Manglish conversations
- Code-switching between English and Malayalam

Use multiple paraphrases of each scenario to improve generalization.

### Evaluation improvements

Expand the held-out evaluation set beyond 20 prompts.

Track separate metrics for:

- Intent understanding
- Correct follow-up questions
- Booking-status hallucinations
- Unsupported price or availability claims
- Duplicate-booking risk
- Malayalam response quality
- Manglish response quality
- Instruction adherence

Keep evaluation prompts separate from training examples to reduce data leakage.

### Training improvements

Keep the training configuration reproducible and change one major factor at a time.

Compare v3, v4 and v5 using the same evaluation prompts, system prompt and decoding settings.

Save checkpoints and record the dataset version, configuration and training step for each experiment.

## 10. Model Storage and Recovery

The base model is stored separately from the LoRA adapter.

The large base-model weights were intentionally excluded from GitHub history. They must be downloaded again or restored from persistent storage when a new Kaggle session is created.

The v4 adapter should also be stored in a persistent location independent of the temporary Kaggle working directory.

For exact training resumption, save the complete trainer checkpoint rather than only the adapter weights.

## 11. Current Status

| Area | Status |
|---|---|
| Dataset preparation | Completed for v4 |
| Deduplication | Completed |
| Tokenization validation | Completed |
| LoRA training | Completed successfully |
| v3/v4 comparison | Completed on 20 prompts |
| Behavioral limitations | Identified |
| Malayalam/Manglish quality | Needs improvement |
| Production readiness | Not established |
| v5 improvements | Recommended |

## Conclusion

Apsara v4 demonstrates that the LoRA training pipeline can complete successfully on the available Kaggle GPU with a relatively small adapter artifact.

However, the evaluation indicates that successful training has not yet translated into clearly improved behavior over v3. False booking confirmations, duplicate-booking risk, unsupported safety claims, and multilingual quality issues remain important areas for improvement.

The next priority is to improve targeted training examples, strengthen evaluation coverage, and verify persistent storage of both the base model and adapter before beginning v5.