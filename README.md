# ✨ Apsara — AI Ride Assistant

<p align="center">
  <img src="apsara/assets/fictional_model.jpeg" alt="Apsara — an AI-inspired character" width="100%">
</p>

<p align="center">
  <strong>An AI-powered ride-sharing assistant for passengers and drivers, focused on natural-language support for shared journeys in Kerala.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-AI%20Fine--Tuning-7C3AED?style=for-the-badge" alt="AI Fine-tuning">
  <img src="https://img.shields.io/badge/Current%20Model-Gemma%203%204B-2563EB?style=for-the-badge" alt="Gemma 3 4B">
  <img src="https://img.shields.io/badge/Training-QLoRA-059669?style=for-the-badge" alt="QLoRA">
  <img src="https://img.shields.io/badge/Status-Experimental-F59E0B?style=for-the-badge" alt="Experimental">
</p>

## About Apsara

**Apsara** is an AI assistant for a ride-sharing application. It is designed to help passengers express their travel requirements and help drivers understand trip scheduling and management through natural-language conversations.

The project explores domain-specific fine-tuning to improve ride-sharing intent understanding, multilingual responses, and conversational behavior, particularly for English, Malayalam, and Manglish.

The current model experiment is **Apsara V7**, based on **Gemma 3 4B Instruct** and fine-tuned using 4-bit quantization and Low-Rank Adaptation (QLoRA) on a Kaggle Tesla T4 GPU.

Earlier experiments used Qwen2.5-1.5B-Instruct. That model remains part of the project's experimental history, but it is no longer the base model used for the V7 experiment.

Apsara remains experimental. Fine-tuning does not by itself provide live ride listings, booking capabilities, driver tracking, or production-ready backend integration.

## Current model: Apsara V7

| Property | Details |
|---|---|
| Version | Apsara V7 |
| Base model | Gemma 3 4B Instruct |
| Model architecture | Gemma 3 conditional-generation model |
| Fine-tuning approach | QLoRA with PEFT LoRA adapters |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | FP32 |
| Training hardware | Kaggle NVIDIA Tesla T4 |
| GPU memory | Approximately 14.56 GB reported |
| Training examples | 153 unique conversations before the train/validation split |
| Training split | 137 examples |
| Validation split | 16 examples |
| Epochs completed | 3 |
| Best recorded training loss | 1.9441 |
| Best recorded validation loss | 1.9224 |
| Adapter output | `apsara/models/apsara_v7/` |

The model was trained using a custom PyTorch training loop with assistant-response loss masking, gradient checkpointing, and LoRA adapters.

The adapter contains approximately 5.95 million trainable parameters, around 0.1382% of the reported 4.306-billion-parameter model.

### Training results

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 4.0198 | 2.7444 |
| 2 | 2.5101 | 2.1332 |
| 3 | 1.9441 | 1.9224 |

Both training and validation loss decreased during the experiment. This indicates that the model learned patterns from the training examples, but it does not establish that its generated responses are accurate, natural, or safe.

Initial manual testing of eight Manglish prompts identified several remaining issues:

- Unnatural Manglish grammar and word endings.
- Repetitive responses that instruct users to check the app.
- Incomplete understanding of booking-status and cancellation requests.
- Loss of route details in some responses.
- Weak handling of fare questions and driver-status requests.

These findings motivate further dataset improvement and systematic evaluation.

> **Evaluation status:** The eight-prompt manual test is an initial qualitative check, not a complete benchmark. The separate 20-example held-out evaluation dataset must remain excluded from training and should be used for a formal comparison of the base model and the fine-tuned adapter.

## Project goals

- Understand passenger and driver requests in natural language.
- Ask relevant follow-up questions when route, date, or departure time is missing.
- Help passengers interpret ride options returned by the application.
- Help drivers understand trip scheduling and management.
- Avoid inventing live availability, prices, seat counts, driver locations, or booking status.
- Explore parameter-efficient fine-tuning using QLoRA and LoRA.
- Improve English, Malayalam, and Manglish response quality.
- Evaluate model versions against held-out examples.
- Eventually connect the assistant to authorized backend APIs for ride search and booking workflows.

## Important design rule

**The language model must not be the source of truth for live application data.**

- Ride listings, prices, available seats, trip status, and booking status must come from the backend.
- The backend must enforce authorization, validate requests, and handle state-changing operations.
- The assistant must not claim a booking or cancellation succeeded unless the backend confirms it.
- The assistant should request confirmation before a state-changing operation when required by the product workflow.
- Private passenger and driver information must only be disclosed when authorized.

## Intended architecture

```text
Passenger / Driver
        |
        v
Ride-Sharing Application
        |
        v
Apsara AI Assistant
Gemma 3 4B Instruct
        |
        v
Apsara V7 LoRA Adapter
        |
        v
Intent understanding and tool selection
        |
        v
Controlled backend API calls
        |
        +--> Ride search
        +--> Trip details and available seats
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

This is the intended architecture. The diagram does not imply that all backend tools, booking operations, or live-data integrations have already been implemented.

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Dataset preparation, model experiments, evaluation |
| Gemma 3 4B Instruct | Current V7 base language model |
| Hugging Face Transformers | Model loading, tokenization, inference, and model execution |
| Hugging Face PEFT | LoRA adapter configuration and fine-tuning |
| bitsandbytes | 4-bit model quantization |
| PyTorch | Model execution and custom training loop |
| Hugging Face Datasets | Dataset loading, splitting, and processing |
| JSONL | Conversational training and evaluation data |
| Kaggle | GPU-based training experiments |
| Git / GitHub | Source control, documentation, and experiment artifacts |
| Qwen2.5-1.5B-Instruct | Earlier model experiment |
| llama.cpp / GGUF | Earlier local inference workflow; separate from the current Gemma adapter |

## Repository structure

The following is a representative structure. Confirm the actual files in the repository before treating it as a complete inventory.

```text
apsara/
├── README.md
├── assets/
│   └── fictional_model.jpeg
├── models/
│   ├── apsara_v7/
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── chat_template.jinja
│   │   ├── manglish_test_results.jsonl
│   │   └── tokenizer and processor configuration files
│   └── qwen/
│       └── ...                         # Earlier experiment artifacts
├── ride_sharing_data/
│   ├── evaluation.jsonl
│   ├── apsara_v7_candidate.jsonl
│   ├── train_manglish_english_v2.jsonl
│   ├── apsara_manglish_english_ride_sharing_seed.jsonl
│   ├── malayalam_manglish_improvement_batch.jsonl
│   └── ...                             # Other dataset and evaluation files
└── prepared_data/
    └── ...                             # Legacy e-commerce experiment data
```

## Model files and loading

### Apsara V7 adapter

The V7 adapter is stored at:

```text
apsara/models/apsara_v7/
```

Important files include:

- `adapter_config.json` — LoRA adapter configuration.
- `adapter_model.safetensors` — trained adapter weights.
- `manglish_test_results.jsonl` — outputs from the initial eight-prompt test.
- Tokenizer and processor configuration files saved during the experiment.

The adapter is **not a complete standalone copy of Gemma 3 4B**. The original base model must be available separately.

The local Kaggle paths used during training were:

```text
/kaggle/working/gemma-3-4b-it
/kaggle/working/apsara_v7_gemma3_4b
```

The first path contains the downloaded base model; the second contains the training output. Kaggle session storage is temporary, so preserve the adapter outside the session before it ends.

### Loading the adapter

Use compatible versions of Transformers, PEFT, PyTorch, and bitsandbytes. The following is an illustrative loading example for a compatible environment:

```python
import torch
from transformers import (
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel

BASE_MODEL = "/path/to/gemma-3-4b-it"
ADAPTER_PATH = "apsara/models/apsara_v7"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

base_model = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    torch_dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
).eval()
```

The original experiment required FP32 compute because an earlier FP16 quantized inference attempt produced non-finite logits. The configuration above reflects the working setup from that experiment; other hardware and software environments may require different settings.

Verify the model's behavior and memory usage before using it in an application.

## Ride-sharing dataset

Ride-sharing conversational data is stored in:

```text
apsara/ride_sharing_data/
```

### V7 training dataset

The V7 candidate dataset was assembled from three source files:

| Source | Loaded examples | Unique examples added |
|---|---:|---:|
| `train_manglish_english_v2.jsonl` | 137 | 137 |
| `apsara_manglish_english_ride_sharing_seed.jsonl` | 44 | 0 |
| `malayalam_manglish_improvement_batch.jsonl` | 30 | 16 |
| **Total** | **211** | **153** |

The exact-conversation deduplication process produced 153 candidate conversations, with zero invalid examples reported.

The candidate file is:

```text
apsara/ride_sharing_data/apsara_v7_candidate.jsonl
```

The 153 examples were split into:

- 137 training examples.
- 16 validation examples.

The separate 20-example `evaluation.jsonl` file was not used by the training loop.

The initial exact-conversation overlap check reported zero matches. That check alone does not establish that there are no semantically similar examples or overlapping user questions with different assistant answers. A stricter question-level leakage check remains useful.

### JSONL conversation format

Each line is an independent JSON object containing a `messages` list.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Enikku nale Aluva ninnu Kochi vare pokanam."
    },
    {
      "role": "assistant",
      "content": "Sheri. Nale eth samayathaanu pokendathu?"
    }
  ]
}
```

Each conversation should be reviewed for factual accuracy, natural language, appropriate follow-up questions, and safe handling of live-data requests.

Training answers should not teach the model to claim that it searched for a ride, confirmed a booking, or retrieved a price unless an actual backend result supports that claim.

### Evaluation data

The held-out evaluation dataset is located at:

```text
apsara/ride_sharing_data/evaluation.jsonl
```

It contains 20 examples intended to test ride-sharing questions, including route and departure-time requests, current availability, and multilingual conversations.

**Do not merge the evaluation dataset into the training dataset.** Doing so would compromise its use as a held-out evaluation set.

## Evaluation and current limitations

The first V7 manual test used eight Manglish prompts covering:

1. Requesting a ride.
2. Asking about ride availability.
3. Checking whether a ride was booked.
4. Providing pickup and destination.
5. Asking about driver location.
6. Cancelling a ride.
7. Requesting a ride home.
8. Asking about a fare.

The model frequently used generic instructions to check the app and produced some unnatural Manglish constructions. These are known limitations, not evidence of production readiness.

### Recommended evaluation criteria

Evaluate the base model and V7 with the same prompts and comparable generation settings.

- **Intent understanding:** Does the response identify the actual request?
- **Slot retention:** Does it preserve pickup, destination, date, and time?
- **Language quality:** Is English, Malayalam, or Manglish natural and consistent with the user's language?
- **Follow-up relevance:** Does it ask only for information that is missing?
- **Factual honesty:** Does it avoid inventing live ride or booking data?
- **Privacy:** Does it protect account-specific information?
- **Action safety:** Does it avoid falsely claiming that a booking or cancellation succeeded?

A decrease in training or validation loss does not guarantee an improvement in these criteria.

## Earlier Qwen experiment

Before V7, the project used Qwen2.5-1.5B-Instruct for initial ride-sharing dataset and LoRA experiments.

Older Qwen model files, GGUF inference workflows, scripts, and documentation may remain in the repository. They represent earlier experiments and should not be confused with the current Gemma 3 4B adapter.

The current V7 adapter is not interchangeable with a Qwen base model. Use the matching Gemma 3 4B base model when loading the V7 adapter.

## Safety and production considerations

Before Apsara is used by real passengers or drivers:

- Implement and test backend ride-search and booking APIs.
- Require authorization for account-specific data and state-changing actions.
- Validate trip and seat availability in the backend.
- Handle duplicate booking attempts and network timeouts safely.
- Never invent a ride, price, seat count, driver location, or booking status.
- Protect passenger and driver contact information.
- Evaluate English, Malayalam, and Manglish conversations.
- Test adversarial and ambiguous requests.
- Log errors without exposing credentials or private data.

Fine-tuning can improve response patterns, but it cannot replace backend validation, authorization, or reliable access to live data.

## Roadmap

- [x] Create a ride-sharing conversational dataset.
- [x] Prepare a separate held-out evaluation dataset.
- [x] Run an initial base-model evaluation.
- [x] Assemble and deduplicate the V7 candidate dataset.
- [x] Split the candidate dataset into training and validation sets.
- [x] Load Gemma 3 4B using 4-bit quantization.
- [x] Attach LoRA adapters and complete a three-epoch training run.
- [x] Record training and validation loss.
- [x] Run an initial eight-prompt Manglish test.
- [ ] Run a stricter training/evaluation leakage audit.
- [ ] Compare V7 against the actual previous fine-tuned checkpoint, if available.
- [ ] Evaluate V7 on all held-out evaluation examples.
- [ ] Improve Manglish naturalness and ride-intent understanding.
- [ ] Expand and review the training dataset.
- [ ] Test the adapter in a clean environment after a Kaggle session restart.
- [ ] Build and test controlled backend tool calling.
- [ ] Implement authorized ride-search and booking workflows.
- [ ] Document measured quality, inference speed, and memory use.

## Repository hygiene

Avoid committing:

- Access tokens, API keys, database credentials, or other secrets.
- Unnecessary virtual environments and caches.
- The complete downloaded base model unless there is a deliberate reason and appropriate storage.
- Private or unlicensed datasets.
- Unnecessary intermediate checkpoints.

The trained V7 adapter may be stored in the repository if it fits GitHub's file-size limits and the applicable model license permits the intended redistribution. If it is too large, use Git LFS or a suitable model-hosting service and link to the artifact from the README.

Review the base model's license and terms before redistributing the adapter or using it commercially.

## Acknowledgements

- [Google Gemma](https://ai.google.dev/gemma) — current V7 base-model family.
- [Qwen](https://huggingface.co/Qwen) — earlier base-model experiments.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — model loading and inference.
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index) — parameter-efficient fine-tuning.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — quantization support.
- [PyTorch](https://pytorch.org/) — model execution and training.
- [Kaggle](https://www.kaggle.com/) — GPU-based experimentation.

---

<p align="center">
  <em>From natural-language requests to smarter shared journeys — one experiment at a time.</em>
</p>
