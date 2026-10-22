# 06 — Model Storage, GitHub Cleanup & Kaggle Session Recovery

## 1. Purpose

This document explains how the Apsara project was backed up to GitHub, why the large base-model files were removed from Git history, and how to restore the project and continue working in a new Kaggle session.

## 2. Current Project Overview

Apsara is a multilingual ride-sharing assistant fine-tuned from `Qwen/Qwen2.5-1.5B-Instruct`.

The training focus includes:
- English ride-sharing conversations
- Malayalam ride-sharing conversations
- Manglish ride-sharing conversations

The project uses LoRA fine-tuning to adapt the base model for ride-sharing customer support.

## 3. GitHub Repository

**Repository:** https://github.com/midhun7ui/apsara

**Branch:** `main`

The repository contains the project code, documentation, datasets and evaluation artifacts that were committed and successfully pushed.

### Why the base model was removed

The Hugging Face base-model weights were approximately 2.94 GB, and the GGUF model was approximately 1.06 GB.

The GitHub push initially failed because the repository's Git payload exceeded the applicable 2 GiB limit.

The following files were removed from Git history:

- `apsara/models/qwen/hf/model.safetensors`
- `apsara/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf`

The `.gitignore` file was updated to prevent these files from being added to Git again.

The Git history was rewritten using `git-filter-repo`, and the cleaned `main` branch was pushed successfully using a controlled force push.

**Important:** Removing these files from Git history does not delete them from the local filesystem. However, local files in Kaggle's temporary working directory may disappear when a session ends.

## 4. Model Storage Strategy

Apsara has three distinct types of model artifacts.

### 4.1 Base model

The base model provides the original pretrained language-model weights.

Model identifier:

`Qwen/Qwen2.5-1.5B-Instruct`

Expected local directory:

`apsara/models/qwen/hf/`

This directory should contain the model weights, tokenizer files, configuration and any other files required by the model loader.

The base model is not stored in the GitHub repository. It must be downloaded again or restored from persistent storage when necessary.

### 4.2 LoRA adapter

The LoRA adapter contains the fine-tuned parameter updates learned during Apsara training.

Example adapter directory:

`apsara/models/qwen/apsara-lora-gpu-v4-20261023/`

Important files may include:

- `adapter_model.safetensors`
- `adapter_config.json`
- Tokenizer files, if required by the workflow

An adapter is not a complete standalone base model. It must be loaded with its compatible base model.

The v4 adapter was approximately 8.34 MB, excluding its tokenizer files. Its permanent backup location must be verified before assuming it can be recovered after a Kaggle session ends.

### 4.3 Training checkpoints

A resumable training checkpoint may contain:

- Model or adapter weights
- Optimizer state
- Learning-rate scheduler state
- Trainer state and current training step
- Random-number-generator state

Saving only `adapter_model.safetensors` is not sufficient to resume the exact training process from its previous step.

## 5. Restoring the Project in Kaggle

Start a new Kaggle notebook session and clone the GitHub repository.

```bash
cd /kaggle/working
git clone https://github.com/midhun7ui/apsara.git
cd apsara
```

The repository can be cloned without downloading the large base-model files.

### Download the base model again

Install the Hugging Face Hub library if necessary:

```python
%pip install -U huggingface_hub
```

Download the model into the expected directory:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir=(
        "/kaggle/working/apsara/"
        "apsara/models/qwen/hf"
    ),
)
```

The download requires internet access and sufficient Kaggle disk space. Verify the model files after the download completes.

If a previously saved private Kaggle Dataset contains the model, attaching that dataset is another option and may avoid downloading it again.

## 6. Restoring and Loading the LoRA Adapter

Before training or inference, check whether the adapter exists in the cloned repository or in persistent storage.

For example:

```python
from pathlib import Path

adapter_dir = Path(
    "/kaggle/working/apsara/apsara/models/qwen/"
    "apsara-lora-gpu-v4-20261023"
)

print("Adapter directory exists:", adapter_dir.exists())

if adapter_dir.exists():
    for path in adapter_dir.iterdir():
        print(path.name)
```

If the adapter is missing, restore it from its verified backup location. Do not assume that a successful base-model download also restores the fine-tuned adapter.

To continue training from the last training step, restore the full checkpoint if available and configure the training script to resume from it.

## 7. Kaggle Session Persistence

Files stored under `/kaggle/working/` should be treated as temporary.

Before ending a session:

1. Push code and documentation changes to GitHub.
2. Save the base model in persistent storage or ensure it can be downloaded again.
3. Back up the LoRA adapter separately.
4. Save a full training checkpoint if exact training resumption is required.
5. Preserve evaluation results and relevant training logs.
6. Verify that each backup is accessible before closing the session.

Saving a Kaggle notebook version preserves the notebook version, but should not be treated as a guaranteed backup of every file in `/kaggle/working/`.

## 8. Git Commands for Future Updates

Check the current status:

```bash
git status
```

Stage selected code and documentation changes:

```bash
git add docs/
git add apsara/ride_sharing_data/
```

Review staged files before committing:

```bash
git status --short
git diff --cached --stat
```

Commit and push:

```bash
git commit -m "Update Apsara documentation and training artifacts"
git push origin main
```

Avoid `git add .` without reviewing the resulting changes, especially when model files or generated checkpoints are present.

If Git history is rewritten again, verify the remote state and review the consequences before using a force push.

## 9. Verification Checklist

Before considering the project safely backed up, verify the following:

- [ ] GitHub contains the latest source code and documentation.
- [ ] The large base-model files are absent from Git history.
- [ ] The base model can be downloaded or restored.
- [ ] The v4 LoRA adapter has a verified persistent backup.
- [ ] The tokenizer and adapter configuration files are available.
- [ ] Evaluation result files are backed up.
- [ ] Training scripts and dataset versions are preserved.
- [ ] A full training checkpoint is saved if exact training resumption is needed.
- [ ] A new Kaggle session can restore the required artifacts.

## 10. Recommended Next Steps

1. Verify the contents of the GitHub repository after the successful push.
2. Confirm that the v4 adapter and its configuration are safely backed up.
3. Restore or download the base model in a new Kaggle session.
4. Run a small inference test using the restored base model and adapter.
5. Continue improving the English, Malayalam and Manglish evaluation results.
6. Save future training checkpoints to persistent storage.

## Summary

GitHub is used for source code, documentation, version-controlled datasets and selected lightweight artifacts. The base model, trained adapters and resumable training checkpoints must be managed separately when they are too large or temporary for ordinary Git storage.

A successful GitHub push protects only the files actually included in the pushed repository. Independent backups are required for any model artifacts excluded from Git.
