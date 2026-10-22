# Apsara — Core Theory

## 1. Base model vs fine-tuned model

A **base/instruction model** already learned language patterns and general instructions during pretraining and instruction tuning. Apsara starts from `Qwen/Qwen2.5-1.5B-Instruct`.

**Fine-tuning** continues training on examples that show the response style and task behavior desired for Apsara.

Fine-tuning does not automatically give a model live information. If the model has no current ride database or tool call, it cannot reliably know which drivers are available now or what a current fare is.

## 2. What is LoRA?

**LoRA** means Low-Rank Adaptation. Instead of updating every parameter in the base model, LoRA trains small additional matrices attached to selected model layers.

Benefits:
- Much lower trainable-parameter count than full fine-tuning.
- Smaller adapter files.
- Lower compute and memory requirements.
- The original base model can remain unchanged.

A LoRA adapter is not usually a complete standalone model. To run it, load the compatible base model and then load the adapter.

## 3. LoRA settings used in this experiment

| Setting | Value | Meaning |
|---|---:|---|
| Rank (`r`) | 8 | Size/capacity of the low-rank update |
| Alpha | 16 | Scaling factor for LoRA updates |
| Dropout | 0.05 | Regularization applied within LoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projection layers adapted |
| Maximum sequence length | 256 | Token limit used for each training example |
| Epochs | 3 | Number of passes over the training dataset |
| Learning rate | `2e-5` | Step size for parameter updates |
| Per-device batch size | 1 | Examples processed per device per micro-batch |
| Gradient accumulation | 4 | Accumulate gradients across four micro-batches before an optimizer step |

These are experiment settings, not universally optimal values.

## 4. JSONL chat data

JSONL means **JSON Lines**: each line is an independent valid JSON object.

A chat-training example often contains a `messages` list with roles such as:
- `system`: behavior rules
- `user`: request or question
- `assistant`: target response

Example shape:

```json
{"messages":[{"role":"system","content":"You are Apsara, a helpful ride-sharing assistant."},{"role":"user","content":"Kochiyilekku oru ride venam."},{"role":"assistant","content":"Sure. Where in Kochi would you like to go?"}]}
```

The exact schema must match what the training script expects.

## 5. Training loss

Loss estimates how different the model's predicted training tokens are from the target tokens under the training objective. Lower training loss can indicate better fit to the training examples, but it does **not** guarantee helpful, correct, or safe responses.

Use held-out evaluation and manual inspection in addition to loss.

## 6. Gradient norm

The gradient norm summarizes the magnitude of gradients used for learning. A `NaN` gradient norm can indicate numerical instability or invalid gradients. It should be investigated rather than ignored.

In this workflow:
- A Trainer smoke test with FP16 Trainer mode reported `grad_norm: nan`.
- A manual one-example FP16 forward/backward test produced finite loss and finite gradients.
- A later Trainer smoke test with `fp16=False` reported finite gradient norms around 1.89 and 1.96.

This suggests the Trainer mixed-precision setting may have contributed, but it does not conclusively establish the root cause.

## 7. FP16 and BF16

FP16 and BF16 are reduced-precision floating-point formats used to reduce memory use and potentially improve speed.

For this run:
- The base model was loaded with `dtype=torch.float16`.
- Trainer settings used `bf16=False` and `fp16=False`.

That means model weights were loaded in FP16, but Trainer's own mixed-precision mode was disabled. Keep the script's printed messages consistent with its actual settings.

## 8. Training vs inference

- **Training:** calculate loss, backpropagate gradients, and update trainable parameters.
- **Inference:** load the base model and adapter, then generate responses without updating parameters.
- **Evaluation:** run fixed prompts through a model and judge the outputs against expected behavior.

Do not treat a successful training run as proof that inference quality is good.
