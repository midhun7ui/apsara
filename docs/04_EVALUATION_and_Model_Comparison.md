# Apsara — Evaluation and Model Comparison

## Why evaluation is essential

Training loss measures fit to training examples. Evaluation checks the model on fixed prompts that were held out from training.

A model can finish training successfully and still:
- Misunderstand Manglish.
- Give generic or irrelevant answers.
- Invent routes, drivers, fares, or booking status.
- Fail to ask for missing pickup or destination details.
- Become worse than the base model on some tasks.

An earlier local CPU LoRA adapter showed poor responses, including invented routes, failure to understand a Manglish request, and generic unrelated advice. This is why the new adapter must be evaluated rather than assumed to be better.

## Evaluation set and intended output

Evaluation file:

```text
/kaggle/working/apsara/apsara/ride_sharing_data/evaluation.jsonl
```

Expected count: 20 examples.

Evaluator script prepared for the new adapter:

```text
/kaggle/working/apsara/apsara/ride_sharing_data/evaluate_lora_kaggle_manglish_v2.py
```

Expected results file:

```text
/kaggle/working/apsara/apsara/ride_sharing_data/lora_kaggle_manglish_v2_results.jsonl
```

**Status recorded:** the evaluator script was prepared, but no evaluation output had yet been reported. Run it before making a quality claim.

## Run the evaluator in Kaggle

```python
import os
import subprocess
from pathlib import Path

project = Path("/kaggle/working/apsara")
script = project / "apsara/ride_sharing_data/evaluate_lora_kaggle_manglish_v2.py"

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

result = subprocess.run(
    ["python", str(script)],
    cwd=project,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=1800,
)

print(result.stdout[-16000:])
print("\\nExit code:", result.returncode)
```

Check that the exit code is `0` and that the expected results file exists.

## Compare models fairly

Use the same held-out prompts and the same generation settings for:
1. Base Qwen model.
2. Earlier local LoRA adapter.
3. New Kaggle LoRA adapter.

Do not compare results generated from different prompt sets or substantially different decoding settings.

Existing comparison script mentioned in the project:

```text
compare_three_models.py
```

It originally expected:
- `base_model_results.jsonl`
- `lora_gpu_test_results.jsonl`
- `lora_cpu_manglish_v2_results.jsonl`

The script must be updated carefully, or a separate comparison should be performed, to include the new Kaggle results without overwriting old outputs.

## Manual scoring rubric

Score each response from 0 to 2 for each dimension:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Intent understanding | Misses intent | Partly understands | Correctly understands |
| Language handling | Fails language | Partly handles it | Natural and clear |
| Relevance | Unrelated | Partly relevant | Directly relevant |
| Clarification | Guesses missing details | Incomplete clarification | Asks for necessary details |
| Truthfulness | Invents facts | Some unsupported detail | Does not fabricate |
| Ride-service boundaries | Claims unsupported live data | Unclear boundaries | Clearly requires live data/tools |

Use scores consistently and keep example notes for failures. A numeric score helps compare versions but should not replace reading the actual responses.

## Essential safety check

Without a live backend/tool response, Apsara must not claim that a ride is booked, that a specific driver is available, or that a fare is current. A language model can phrase a request and ask clarifying questions; a trusted system must provide live ride facts.
