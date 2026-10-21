# ✨ Apsara — AI Ride Assistant

<p align="center">
  <img src="apsara/assets/fictional_model.jpeg" alt="Apsara — an AI-inspired character" width="100%">
</p>

<p align="center">
  <strong>An AI-powered ride-sharing assistant for passengers and drivers, focused on natural-language support for shared journeys in Kerala.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-AI%20Fine--Tuning-7C3AED?style=for-the-badge" alt="AI Fine-tuning">
  <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B-2563EB?style=for-the-badge" alt="Qwen2.5-1.5B">
  <img src="https://img.shields.io/badge/Domain-Ride%20Sharing-059669?style=for-the-badge" alt="Ride Sharing">
  <img src="https://img.shields.io/badge/Status-In%20Development-F59E0B?style=for-the-badge" alt="In Development">
</p>

## About Apsara

**Apsara** is an AI assistant for a ride-sharing application. It is intended to help passengers find rides and help drivers understand how to schedule and manage trips.

The project uses **Qwen2.5-1.5B-Instruct** as its initial language model. The current work focuses on preparing ride-sharing training data, evaluating the base model, and preparing an initial LoRA fine-tuning experiment.

Apsara is still in development. A fine-tuned model, live ride API integration, and production-ready booking workflows have **not** been established yet.

## Project goals

- Understand passenger and driver requests in natural language.
- Ask for missing route, date, and departure-time information.
- Help passengers compare ride options returned by the application.
- Help drivers understand trip scheduling and management.
- Avoid inventing live availability, prices, seat counts, driver locations, or booking status.
- Explore parameter-efficient fine-tuning using LoRA.
- Evaluate English, Malayalam, and Manglish responses.
- Eventually connect the assistant to controlled backend APIs for ride search and booking workflows.

## Important design rule

The language model must not be treated as the source of truth for live application data.

- Ride listings, prices, available seats, trip status, and booking status must come from the backend.
- The backend must enforce authorization, validate requests, and handle state-changing operations.
- The assistant should ask for confirmation before submitting a booking or cancellation when required by the product workflow.
- It must not disclose another user's private contact information without proper authorization.

## Intended architecture

```text
Passenger / Driver
        |
        v
Ride-Sharing Application
        |
        v
Apsara AI Assistant
Qwen2.5-1.5B-Instruct + optional LoRA adapter
        |
        v
Intent understanding and tool selection
        |
        v
Controlled backend API calls
        |
        +--> Ride search
        +--> Trip details / seats / contribution
        +--> Booking status
        +--> Authorized booking or cancellation
        |
        v
Application database
        |
        v
Verified API result
        |
        v
Natural-language response
```

This is the **intended architecture**. It does not mean all API tools or booking flows have already been implemented.

## Technology stack

| Technology | Purpose / current use |
|---|---|
| Python | Dataset scripts, model experiments, and evaluation |
| Qwen2.5-1.5B-Instruct | Initial base language model |
| Hugging Face Transformers | Loading the model and running inference/training experiments |
| Hugging Face PEFT | LoRA adapter configuration and fine-tuning |
| Hugging Face Datasets | Available for dataset workflows |
| JSONL | Conversational training and evaluation data |
| llama.cpp | Local inference using the quantized GGUF model |
| Vulkan | GPU acceleration for supported `llama.cpp` inference builds |
| PyTorch | Model execution and training |
| Git / GitHub | Version control and documentation |

The current Transformers/PyTorch environment reports that CUDA is unavailable. CUDA is NVIDIA-specific; the laptop has AMD GPUs, so AMD GPU training needs a separately verified compatible compute stack. Local GPU inference with Vulkan does not prove that PyTorch GPU training works.

## Repository structure

The following tree documents the files created or discussed in the current development workflow. Some older project files may exist in addition to these.

```text
LLM-MODEL-E-COMMERCE/
├── README.md
├── env/                              # Existing project Python environment
└── apsara/
    ├── assets/
    │   └── fictional_model.jpeg      # README image
    │
    ├── models/
    │   └── qwen/
    │       ├── hf/                   # Downloaded Hugging Face model files
    │       │   ├── config.json
    │       │   ├── tokenizer files
    │       │   └── model.safetensors
    │       ├── qwen2.5-1.5b-instruct-q4_k_m.gguf
    │       └── apsara-lora-v1/        # Intended output for LoRA adapter
    │
    ├── ride_sharing_data/
    │   ├── train.jsonl
    │   ├── train_before_batch_2.jsonl
    │   ├── generated_review.jsonl
    │   ├── generated_batch_2.jsonl
    │   ├── train_before_improvements.jsonl
    │   ├── improvement_review.jsonl
    │   ├── evaluation.jsonl
    │   ├── base_model_results.jsonl
    │   ├── generate_dataset.py
    │   ├── generate_batch_2.py
    │   ├── generate_improvement_batch.py
    │   ├── merge_batch_2.py
    │   ├── merge_improvements.py
    │   ├── validate_ride_sharing_data.py
    │   ├── create_evaluation.py
    │   ├── evaluate_base_model.py
    │   └── train_lora.py
    │
    ├── prepared_data/                # Legacy e-commerce experiment data
    │   ├── train.jsonl
    │   ├── test.jsonl
    │   └── filtered_experiment/
    │       └── train.jsonl
    │
    ├── prepare_dataset.py            # Legacy dataset preparation
    ├── inspect_dataset.py             # Legacy dataset inspection
    ├── audit_dataset.py              # Legacy dataset audit
    ├── filter_dataset.py             # Legacy dataset filtering
    ├── validate_filtered.py          # Legacy dataset validation
    ├── placeholder_audit.py           # Legacy placeholder audit
    ├── download_datasets.py          # Legacy dataset download helper
    └── llm-test.py                   # Earlier local model test
```

**Note:** The tree lists known files from the current workflow. Confirm with `find apsara -maxdepth 4 -type f` before treating it as an exact inventory. The `apsara-lora-v1/` directory is the configured output location; it may not exist until training completes successfully.

## Model files

### Hugging Face model

Location:

```text
apsara/models/qwen/hf/
```

This directory contains the local Hugging Face model and tokenizer files used by the Transformers scripts. The base model is loaded from this directory.

### GGUF model

Location:

```text
apsara/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

This is the quantized GGUF model used for local `llama.cpp` inference. It is separate from the Hugging Face model used by the current Transformers training script.

Example inference command (the Vulkan device name can differ by installation):

```bash
llama-cli \
  -m apsara/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --device Vulkan1 \
  --gpu-layers 99 \
  --ctx-size 2048
```

This runs inference; it does **not** fine-tune the model or connect it to live ride APIs.

## Ride-sharing dataset

All new ride-sharing conversational data is stored in:

```text
apsara/ride_sharing_data/
```

### Current dataset status

At the time this README was updated:

- `train.jsonl` contains **64 examples**.
- `evaluation.jsonl` contains **20 examples**.
- A check found **0 exact user-question overlaps** between the training and evaluation files.
- The training dataset structure validator reported **0 errors**.
- `base_model_results.jsonl` stores the base model's generated answers to the evaluation questions.

The evaluation set must remain separate from training. Do not merge `evaluation.jsonl` into `train.jsonl`; otherwise, the evaluation would no longer be a clean held-out test.

### JSONL conversation format

Each line in `train.jsonl` is an independent JSON object with a `messages` list. The current dataset examples use a user message and an assistant response:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I need a ride from Aluva to Kakkanad tomorrow morning."
    },
    {
      "role": "assistant",
      "content": "What departure time would you prefer? I'll need current ride listings to confirm availability."
    }
  ]
}
```

Each JSON object must be on a single line in the actual JSONL file. The example above is formatted across multiple lines only for readability.

Training answers should be accurate and diverse. Avoid training the model to claim that it has searched for a ride, confirmed a booking, or retrieved a price unless a real backend result supports that claim.

## Ride-sharing data files and scripts

| File | Purpose |
|---|---|
| `train.jsonl` | Current ride-sharing fine-tuning dataset. It currently contains 64 examples. |
| `generated_review.jsonl` | Earlier generated/review batch retained as a source/reference file. |
| `generated_batch_2.jsonl` | Second generated batch, containing 31 examples in the current workflow. |
| `improvement_review.jsonl` | New 15-example improvement batch targeting hallucinations, privacy, booking-status handling, and local-language conversations. |
| `train_before_batch_2.jsonl` | Backup of the 18-example training dataset before batch 2 was merged. |
| `train_before_improvements.jsonl` | Backup of the 49-example training dataset before the 15 improvement examples were merged. |
| `evaluation.jsonl` | Held-out evaluation questions. Keep this separate from training. |
| `base_model_results.jsonl` | Base model outputs for the 20 evaluation questions, including expected and generated answers. |
| `generate_dataset.py` | Generates the initial review examples. |
| `generate_batch_2.py` | Generates the second batch of ride-sharing examples. |
| `generate_improvement_batch.py` | Creates the 15-example improvement batch. |
| `merge_batch_2.py` | Appends unseen questions from batch 2 and makes a backup before merging. |
| `merge_improvements.py` | Validates and merges improvement examples while skipping exact duplicate user questions; makes a backup before writing. |
| `validate_ride_sharing_data.py` | Checks the structure of `train.jsonl`. |
| `create_evaluation.py` | Creates the separate evaluation set. |
| `evaluate_base_model.py` | Runs the base model on evaluation questions and saves the results. |
| `train_lora.py` | Initial CPU-oriented LoRA training experiment; syntax and data-preparation checks have passed, but successful training has not yet been confirmed. |

The generated/review and backup files are kept for traceability. They are not all separate training sets. The file used by the current training script is `train.jsonl`.

## Dataset commands

Run commands from the repository root.

### Validate the training dataset

```bash
python apsara/ride_sharing_data/validate_ride_sharing_data.py
```

Expected current result:

```text
Total examples: 64
Errors: 0
Dataset structure validation passed.
```

### Check training/evaluation overlap

```bash
python -c "import json; from pathlib import Path; b=Path('apsara/ride_sharing_data'); load=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; train=load(b/'train.jsonl'); ev=load(b/'evaluation.jsonl'); getq=lambda r: next(m['content'].strip().casefold() for m in r['messages'] if m['role']=='user'); tq={getq(r) for r in train}; overlap=[getq(r) for r in ev if getq(r) in tq]; print('Training examples:',len(train)); print('Evaluation examples:',len(ev)); print('Overlapping questions:',len(overlap)); [print('-',q) for q in overlap]"
```

The most recent run reported 64 training examples, 20 evaluation examples, and 0 exact question overlaps. This check only catches exact normalized user-question matches; it does not guarantee that the sets contain no semantically similar examples.

## Base model evaluation

Run:

```bash
python apsara/ride_sharing_data/evaluate_base_model.py
```

The script evaluates the base Qwen model using `evaluation.jsonl` and writes:

```text
apsara/ride_sharing_data/base_model_results.jsonl
```

The first baseline run showed issues that need improvement, including invented ride information, weak Malayalam/Manglish responses, and an unsafe response involving another passenger's contact details. Some responses handled missing details or booking confirmation more appropriately.

Use the same evaluation questions and comparable generation settings when evaluating a fine-tuned model. Compare outputs manually against explicit criteria such as relevance, factual honesty, privacy, safe booking behavior, and language quality. A model's generated answer should not be treated as correct merely because it is fluent.

## LoRA training experiment

Training script:

```text
apsara/ride_sharing_data/train_lora.py
```

Configured base model:

```text
apsara/models/qwen/hf/
```

Configured adapter output:

```text
apsara/models/qwen/apsara-lora-v1/
```

The script uses Hugging Face Transformers `Trainer` and PEFT LoRA. Its current settings include:

- CPU training (`use_cpu=True`)
- FP32 base model loading
- LoRA rank 8 and alpha 16
- Batch size 1
- Gradient accumulation steps 4
- 3 epochs
- Learning rate `1e-4`
- Gradient checkpointing
- Assistant-response-only loss masking

The script has passed a Python syntax check and a dataset-preparation check. **A completed training run and a successful adapter load have not yet been confirmed.** CPU training may be slow and memory-intensive on this laptop.

To check syntax:

```bash
python -m py_compile apsara/ride_sharing_data/train_lora.py
```

To start the current experiment, after reviewing the configuration and ensuring sufficient memory:

```bash
python apsara/ride_sharing_data/train_lora.py
```

If training completes, the adapter is intended to be saved under `apsara/models/qwen/apsara-lora-v1/`. The original base model is not supposed to be overwritten.

### Training limitations

- Sixty-four examples are enough only for an initial experiment, not for establishing production reliability.
- Training loss decreasing does not prove that answer quality improves.
- The current script is configured for CPU; it does not automatically use the AMD GPUs.
- Evaluate the resulting adapter on the same held-out questions before deciding whether to continue training.

## AMD GPU investigation

The laptop has two AMD GPU devices detected by `lspci`:

- Discrete Radeon RX 6500M-family device: PCI ID `1002:743f`, architecture detected as `gfx1034`.
- Integrated Cezanne/Radeon Vega device: PCI ID `1002:1638`, architecture detected as `gfx90c`.

The Linux system exposes `/dev/kfd` and `/dev/dri/renderD*`, but those device nodes alone do not prove that ROCm/PyTorch compute works.

At the time of writing:

- `/opt/rocm` was not present.
- `rocminfo` was not installed/available as a command.
- `rocm-bootstrap` 0.3.0 was installed in a separate environment at `~/rocm-test`.
- Running `rocm-bootstrap-detect` printed `gfx1034` and `gfx90c`.
- PyTorch in the project's existing environment reported `torch.cuda.is_available() == False`.
- A compatible PyTorch GPU runtime has **not** yet been installed or verified.
- No GPU fine-tuning run has been confirmed.

Do not infer that GPU training works merely because the GPU architecture is detected. Any experimental ROCm/TheRock package should be tested in an isolated environment first, without replacing the project's existing PyTorch setup or system graphics drivers.

## Legacy e-commerce dataset

The following directory contains data from an earlier e-commerce chatbot experiment:

```text
apsara/prepared_data/
├── train.jsonl
├── test.jsonl
└── filtered_experiment/
    └── train.jsonl
```

These files are retained for reference and are **not the Apsara ride-sharing training dataset**. Do not merge them into `apsara/ride_sharing_data/train.jsonl`. Review the original dataset's license and terms before redistribution or commercial use.

## Safety and production considerations

Before Apsara is used by real passengers or drivers:

- Implement backend ride-search and booking APIs.
- Require authorization for account-specific data and state-changing actions.
- Validate current trip and seat availability at the backend.
- Avoid duplicate bookings after timeouts by checking booking status before retrying.
- Never invent a ride, price, seat count, driver location, or booking status.
- Protect passenger contact details and other private information.
- Test English, Malayalam, and Manglish conversations.
- Evaluate both normal and adversarial requests.
- Log errors without leaking credentials or private data.

Fine-tuning can help shape response patterns, but it does not replace backend validation, permissions, or live data retrieval.

## Roadmap

- [x] Select Qwen2.5-1.5B-Instruct as the initial base model.
- [x] Download the Hugging Face model and retain a quantized GGUF model for inference.
- [x] Create a separate ride-sharing dataset directory.
- [x] Expand the ride-sharing training dataset to 64 examples.
- [x] Validate the current training dataset structure.
- [x] Create a separate 20-question evaluation set.
- [x] Confirm zero exact user-question overlaps between training and evaluation.
- [x] Run baseline evaluation and save its results.
- [x] Create an initial LoRA training script and validate its syntax/data preparation.
- [x] Detect local AMD GPU architecture names in an isolated ROCm test environment.
- [ ] Complete a GPU runtime compatibility test, if feasible.
- [ ] Run the first successful LoRA training experiment.
- [ ] Load and test the saved adapter.
- [ ] Compare base and fine-tuned responses on the held-out evaluation set.
- [ ] Improve dataset quality and add more diverse reviewed examples.
- [ ] Build and test controlled backend tool calling.
- [ ] Implement authorized ride-search and booking workflows.
- [ ] Document measured quality, speed, and memory use.

## Repository hygiene

Avoid committing the following unless there is a deliberate reason:

- Large model weights and checkpoints
- Virtual environments (`env/`, `rocm-test/`)
- Private datasets or user information
- API keys, access tokens, database credentials, or other secrets

The `rocm-test` environment is located in the home directory (`~/rocm-test`), outside this repository.

## Acknowledgements

- [Qwen](https://huggingface.co/Qwen) — base language model
- [Hugging Face](https://huggingface.co/) — Transformers and model tooling
- [PEFT](https://huggingface.co/docs/peft/index) — parameter-efficient fine-tuning
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — local GGUF inference
- [TheRock](https://github.com/ROCm/TheRock) — experimental AMD GPU compute ecosystem

---

<p align="center">
  <em>From natural-language requests to smarter shared journeys — one experiment at a time.</em>
</p>
