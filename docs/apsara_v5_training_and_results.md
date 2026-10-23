# Apsara v5 — Training and Evaluation Report

**Project:** Apsara  
**Model family:** Qwen2.5-1.5B-Instruct  
**Fine-tuning method:** LoRA (Low-Rank Adaptation)  
**Version:** v5  
**Training date:** 24 October 2026  
**Status:** Training completed; evaluation identified significant quality issues

---

## 1. Overview

Apsara is a ride-sharing customer-service assistant intended to handle rider conversations. The v5 experiment fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` with LoRA.

This release is an experimental checkpoint, not a production-ready model. Training completed successfully, but the evaluation described below found failures in trip-detail preservation, Manglish understanding, and avoidance of unsupported claims.

## 2. Training environment

| Item | Value |
|---|---|
| Compute environment | Kaggle notebook |
| GPU | NVIDIA Tesla T4 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuning method | LoRA |
| Dataset used | `train_combined_v5.jsonl` |
| Dataset size | 193 unique records |
| Dataset composition | 180 existing records + 13 refined date-related records |
| Maximum sequence length | 512 tokens |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 4 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Training steps reported | 150 |
| Elapsed training time | Approximately 6 minutes 35 seconds |
| Peak GPU memory reported | 3.61 GB |

**Reproducibility note:** These are the settings and results recorded for the v5 run. The exact package environment, random seed, and all hardware/software details should also be captured if strict reproducibility is required.

## 3. Dataset and preprocessing

The v5 training file was `train_combined_v5.jsonl`, containing 193 unique records. It was built from the 180-record base dataset plus 13 refined date-related records.

The training script, `train_lora_gpu_v2.py`, was updated to prepare one training row per assistant turn:
- Prior conversation turns are used as the prompt context.
- Prompt tokens are masked from the loss using label value `-100`.
- The assistant answer tokens are the intended training target.
- Samples exceeding the configured maximum sequence length are skipped rather than truncated.
- System messages are included in the preceding prompt context.

The script backup was kept separately as `train_lora_gpu_v2.py.backup` during development. Do not treat a backup as part of the required inference artifacts.

## 4. Training outcome

The v5 training run completed successfully.

| Metric | Observed result |
|---|---|
| Training progress | `150/150` steps |
| Elapsed time | `06:35` |
| Reported step time | Approximately `2.64 s/step` |
| Peak GPU memory | `3.61 GB` |
| Adapter output | `apsara-lora-gpu-v5-20261024` |
| Adapter files | `adapter_config.json`, `adapter_model.safetensors` |

Successful training means the optimization run finished and saved an adapter. It does **not** by itself demonstrate that the model responds accurately or is suitable for production.

## 5. Qualitative evaluation

A small manual evaluation compared v4 and v5 on five English and mixed-language scenarios, followed by a separate ten-prompt Manglish diagnostic test for v5.

These were qualitative spot checks, not a statistically rigorous benchmark.

### 5.1 Five-scenario comparison: v4 vs v5

The prompts covered:
1. A ride request with origin, destination, date, and time.
2. A request for a ride with missing details.
3. A multi-turn change to the pickup location.
4. A ride request written in Manglish.
5. A request to book immediately and ask whether it was confirmed.

**Observed results:**
- **Trip-detail collection:** Both v4 and v5 asked for the destination again in a prompt where it was already provided.
- **Missing details:** Both responses were not sufficiently focused. v5 produced a longer list of suggestions and reached the generation token limit.
- **Multi-turn correction:** Both correctly changed the pickup to Aluva while preserving the destination, date, and time in the tested conversation.
- **Manglish request:** Both failed to understand the test prompt reliably.
- **Booking confirmation:** Both avoided falsely stating that a booking was confirmed, but the responses were generic and did not guide the user effectively.

**Comparison conclusion:** The five-prompt test did not establish a clear overall improvement from v4 to v5. The successful multi-turn correction was shared by both versions.

### 5.2 Ten-prompt Manglish diagnostic test: v5

The diagnostic prompts tested ride requests, pickup corrections, fare questions, booking status, date/time details, underspecified requests, driver location, route availability, airport pickup details, and correction of a previously stated pickup.

**Result:** `0/10` responses were fully satisfactory under the manual review criteria.

Observed failure patterns included:
- Inventing driver availability, distance, travel time, or other trip details.
- Losing or changing pickup and destination information.
- Asking for details that were already supplied.
- Producing incoherent or irrelevant responses.
- Failing to answer fare and booking-status questions appropriately.

This result applies only to this small diagnostic set; it is not a formal estimate of general model accuracy. Nevertheless, it identifies serious issues that need to be fixed before deployment.

## 6. Current limitations

The v5 experiment indicates that further training and evaluation are required in the following areas:

1. **Entity and trip-detail preservation:** Keep pickup, destination, date, and time consistent across turns.
2. **Clarification behavior:** Ask only for information that is genuinely missing.
3. **Hallucination control:** Do not invent fares, driver availability, route distance, travel duration, or booking status.
4. **Booking safety:** Never claim a ride is booked or confirmed unless a connected booking system verifies it.
5. **Multilingual robustness:** Manglish performance is not reliable enough for release.
6. **Response quality:** Avoid repetitive, incoherent, irrelevant, or unnecessarily long answers.

The model should not claim live ride availability or access to booking functionality unless those capabilities are actually connected and verified by the application.

## 7. Artifact locations

In the Kaggle project, the v5 adapter was saved at:

```text
apsara/models/qwen/apsara-lora-gpu-v5-20261024/
```

Expected adapter files include:

```text
adapter_config.json
adapter_model.safetensors
```

The training dataset and script were located at:

```text
apsara/ride_sharing_data/train_combined_v5.jsonl
apsara/ride_sharing_data/train_lora_gpu_v2.py
```

The repository may also include tokenizer-related files in the adapter output directory. The base Qwen model is separate from the LoRA adapter and must be available when loading it, unless the model is explicitly merged and exported.

## 8. Recommended next steps

The next planned iteration is **v6**, with English quality as the first priority.

1. Audit the existing English dataset and identify duplicates and coverage gaps.
2. Build a larger, higher-quality English-only dataset with realistic multi-turn conversations.
3. Create a held-out evaluation set that is not used for training.
4. Evaluate detail preservation, clarification behavior, date/time handling, and hallucination resistance.
5. Train v6 from the base Qwen model rather than assuming v5 provides a reliable foundation.
6. Compare v6 against v4 and v5 using the same prompts and consistent evaluation criteria.
7. Add Manglish training after English behavior becomes reliable, then run regression tests for both languages.

A proposed target is 500 high-quality English conversations, distributed across ride search, multi-turn corrections, missing information, booking safety, dates and times, cancellations/payments, and general help. This is a future target, not the size of the current v5 dataset.

## 9. Summary

- **Training:** Completed successfully on a Kaggle Tesla T4.
- **Artifact:** LoRA adapter saved for v5.
- **Dataset:** 193 unique records in `train_combined_v5.jsonl`.
- **Evaluation:** Small manual tests found significant response-quality issues.
- **Release recommendation:** Keep v5 as an experimental checkpoint; do not label it production-ready.
- **Next milestone:** Improve English behavior, build a held-out evaluation set, and train/evaluate v6 before expanding Manglish support.
