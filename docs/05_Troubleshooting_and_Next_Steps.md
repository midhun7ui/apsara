# Apsara — Troubleshooting and Next Steps

## Issue: `grad_norm: nan`

A NaN gradient norm is a warning that training may be numerically unstable.

What happened in this workflow:
- One Trainer smoke test with FP16 Trainer mode reported `grad_norm: nan`.
- A manual FP16 forward/backward test had finite loss and gradients.
- A later Trainer smoke test with `fp16=False` produced finite gradient norms.

What to do:
1. Stop before launching a long run if the smoke test produces NaNs.
2. Verify loss and gradients are finite.
3. Check the actual model dtype and Trainer precision settings.
4. Try a minimal run with Trainer mixed precision disabled.
5. Inspect logs and library versions.
6. Confirm that the full run has finite loss and gradient norms.

Do not claim a proven root cause unless a controlled test establishes it.

## Issue: confusing precision logs

The training script printed a BF16 message even though the model was loaded with `dtype=torch.float16`.

Keep configuration messages synchronized with the actual code. In experiment notes, distinguish:
- Model loading dtype.
- Trainer `fp16` option.
- Trainer `bf16` option.

These settings are related but not identical.

## Issue: model output is poor after training

Training completion does not guarantee good responses. Possible causes include:
- Too few or too repetitive examples.
- Inconsistent target responses.
- Dataset examples that teach the wrong behavior.
- Poor coverage of spelling and language variations.
- Overfitting to training examples.
- Evaluation prompts that expose a capability gap.
- A model that needs a live tool for factual information.

Recommended process:
1. Run the held-out evaluator.
2. Read every generated answer.
3. Categorize failures (intent, language, hallucination, missing details, irrelevant answer).
4. Improve the examples that target those failure types.
5. Retrain a new version with a new output directory.
6. Re-evaluate on the same held-out set, and ideally add a second unseen test set.
7. Keep old adapters and result files for comparison.

## Issue: the adapter does not load

Check:
- The base model matches the model used for training.
- The adapter path exists.
- `adapter_config.json` and `adapter_model.safetensors` exist.
- PEFT and Transformers versions are compatible.
- The tokenizer and prompt format match the evaluator's expectations.

An adapter is not necessarily a full model checkpoint; it normally requires the base model.

## Next steps in order

- [ ] Run `evaluate_lora_kaggle_manglish_v2.py`.
- [ ] Confirm the results JSONL file exists.
- [ ] Inspect all 20 responses.
- [ ] Compare the base model, previous adapter, and new Kaggle adapter on identical prompts.
- [ ] Record failure categories and score each response.
- [ ] Improve dataset examples based on observed failures.
- [ ] Train a new adapter to a separate output path.
- [ ] Repeat evaluation before choosing a model.
- [ ] Plan backend/API integration for real-time ride data.
- [ ] Test tool failures, missing data, and unsafe or unsupported requests.

## Final study summary

Remember the difference between these outcomes:

- **Data validation passed:** the file format and basic structure look valid.
- **Training passed:** optimization completed and an adapter was saved.
- **Evaluation passed:** the outputs meet defined quality and safety criteria.
- **Production readiness:** the model, backend tools, monitoring, privacy, and failure handling have been tested together.

These are four different milestones.
