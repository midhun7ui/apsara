# Apsara — Dataset Preparation and Validation

## Dataset files

Training file:

```text
apsara/ride_sharing_data/train_manglish_english_v2.jsonl
```

Held-out evaluation file:

```text
apsara/ride_sharing_data/evaluation.jsonl
```

Current recorded sizes:
- Training examples: **137**
- Evaluation examples: **20**

## Why separate training and evaluation data?

The training set teaches the model patterns. The evaluation set checks behavior on examples it did not train on.

If evaluation examples are included in training, the score becomes less meaningful because the model may have seen the answers already.

Keep the evaluation file separate and avoid using it as extra training data during the same comparison.

## Validation checks

Before training, verify:
1. Every non-empty line is valid JSON.
2. Each example has the expected structure.
3. The `messages` field is present and contains usable messages.
4. Message roles and content types match the training script's assumptions.
5. No examples have empty message content.
6. The train and evaluation sets are separate.
7. The data covers realistic English, Malayalam, and Manglish phrasing.
8. Responses do not invent real-time availability, fares, or confirmed bookings.

The previous validation run reported 137 training examples, zero empty-message examples, and no invalid JSON lines.

## Quality matters more than raw count

A small dataset can demonstrate the workflow, but it may not cover enough variation for a reliable production assistant. Inspect:
- Different spellings of Manglish words.
- Malayalam script and Latin-script Malayalam, if both are supported.
- Short, incomplete, and ambiguous rider requests.
- Pickup and destination clarification.
- Cancellation, booking changes, pricing questions, and safety issues.
- Requests for information the model cannot know without a live tool.

## Example of a safe training target

User:
> Kochiyilekku oru ride venam, njan Perumbavoor aanu.

A good assistant should ask for missing trip details or explain that a live booking requires the ride service. It should not invent a driver, pickup time, fare, or confirmed route.

## Suggested dataset review checklist

- [ ] User messages sound natural.
- [ ] Assistant replies answer the request directly.
- [ ] Missing details are clarified instead of guessed.
- [ ] No fabricated live ride data.
- [ ] Similar examples use consistent behavior.
- [ ] Evaluation examples are not duplicated in training.
- [ ] Examples cover both common and difficult prompts.
