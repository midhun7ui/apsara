# Apsara — Kaggle LoRA Training Workflow

## Environment recorded for the run

- GPU: Tesla T4, approximately 14.56 GiB per GPU.
- Two T4 GPUs were present, but the subprocess was constrained to one GPU with `CUDA_VISIBLE_DEVICES=0`.
- Installed versions reported:
  - PyTorch `2.11.0+cu128`
  - Transformers `5.20.2`
  - PEFT `0.20.0`
  - Accelerate `1.14.0`
- `torchao` was uninstalled because an older version caused a PEFT import error; its absence was verified.

Version combinations can change. If reproducing the run later, record the actual versions in that session.

## Training script

```text
/kaggle/working/apsara/apsara/ride_sharing_data/train_lora_kaggle.py
```

Base model directory:

```text
/kaggle/working/apsara/apsara/models/qwen/hf
```

Output adapter directory:

```text
/kaggle/working/apsara/apsara/models/qwen/apsara-lora-kaggle-manglish-v2-20261022
```

## Training settings

| Parameter | Value |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | 256 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | `2e-5` |
| Optimizer | `adamw_torch` |
| Gradient checkpointing | Enabled |
| Trainer `bf16` | `False` |
| Trainer `fp16` | `False` |
| Model loading dtype | `torch.float16` |

The script contained a print statement saying it was loading the base model in BF16, while the actual model load used FP16. Update that message so logs accurately describe the configuration.

## Why run a smoke test first?

A smoke test uses a tiny amount of training to check that:
- The model and tokenizer load.
- The data can be processed.
- The loss is computed.
- Backpropagation works.
- The adapter can be saved.

This catches errors before spending time on the full run.

## Smoke-test history

1. First Trainer one-step smoke test: loss about `3.3`, gradient norm was `NaN`.
2. Manual one-example FP16 forward/backward test: loss `3.9273`, finite loss and finite gradients; maximum absolute gradient was approximately `0.371699`.
3. Second Trainer smoke test with `fp16=False`: loss about `3.3`, gradient norms around `1.89` and `1.96`, runtime about 1.96 seconds, peak GPU memory about 3.07 GB, exit code 0.

The evidence suggests the Trainer mixed-precision mode may have contributed to the NaN result, but the exact root cause was not conclusively isolated.

## Full training result

The full run completed successfully:
- 105 / 105 optimizer steps.
- 3 epochs.
- Runtime approximately 131.6 seconds.
- Reported final training loss: `3.224`.
- Gradient norms shown in the run were finite.
- Peak GPU memory: approximately 3.38 GB.
- Exit code: 0.
- Adapter saved successfully.

Adapter files include `adapter_model.safetensors` (about 8.34 MB) and `adapter_config.json`, along with tokenizer files.

## What this proves — and what it does not

It proves that the training script completed and produced an adapter file.

It does not prove that:
- Manglish understanding improved.
- The assistant avoids hallucinations.
- Responses are better than the base model.
- The model is ready for production.

Those claims require held-out evaluation and response inspection.
