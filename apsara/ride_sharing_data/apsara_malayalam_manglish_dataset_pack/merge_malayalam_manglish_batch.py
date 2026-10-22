import json
import shutil
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train.jsonl"
EVAL = BASE / "evaluation.jsonl"
BATCH = BASE / "malayalam_manglish_improvement_batch.jsonl"


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def user_question(row):
    users = [m.get("content", "") for m in row.get("messages", [])
             if m.get("role") == "user"]
    if not users:
        raise ValueError("Every example must contain a user message.")
    return " ".join(users).strip().casefold()


def validate_row(row, label):
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{label}: missing messages list")
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{label}: requires user and assistant messages")
    for m in messages:
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            raise ValueError(f"{label}: message content must be non-empty text")


def main():
    for path in (TRAIN, EVAL, BATCH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = load_jsonl(TRAIN)
    evaluation = load_jsonl(EVAL)
    batch = load_jsonl(BATCH)

    for i, row in enumerate(train, 1):
        validate_row(row, f"train row {i}")
    for i, row in enumerate(evaluation, 1):
        validate_row(row, f"evaluation row {i}")
    for i, row in enumerate(batch, 1):
        validate_row(row, f"batch row {i}")

    eval_questions = {user_question(row) for row in evaluation}
    train_questions = {user_question(row) for row in train}
    accepted = []
    skipped_train_duplicates = []
    skipped_eval_overlap = []

    for i, row in enumerate(batch, 1):
        q = user_question(row)
        if q in eval_questions:
            skipped_eval_overlap.append(i)
        elif q in train_questions:
            skipped_train_duplicates.append(i)
        else:
            accepted.append(row)
            train_questions.add(q)

    print(f"Current training examples: {len(train)}")
    print(f"Evaluation examples: {len(evaluation)}")
    print(f"Proposed batch examples: {len(batch)}")
    print(f"Accepted additions: {len(accepted)}")
    print(f"Skipped duplicate training questions: {len(skipped_train_duplicates)}")
    print(f"Skipped exact evaluation overlaps: {len(skipped_eval_overlap)}")

    if not accepted:
        print("No new examples to merge; dataset left unchanged.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TRAIN.with_name(f"train_before_malayalam_manglish_{stamp}.jsonl")
    shutil.copy2(TRAIN, backup)
    with TRAIN.open("a", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_train = load_jsonl(TRAIN)
    final_questions = {user_question(row) for row in final_train}
    exact_overlap = final_questions & eval_questions

    print(f"Backup created: {backup}")
    print(f"Updated training examples: {len(final_train)}")
    print(f"Unique normalized training questions: {len(final_questions)}")
    print(f"Exact normalized train/evaluation overlaps: {len(exact_overlap)}")
    if exact_overlap:
        print("WARNING: exact evaluation overlap found:")
        for q in sorted(exact_overlap):
            print("-", q)
        raise SystemExit("Review the overlap before training.")

    print("Merge complete. evaluation.jsonl was not modified.")
    print("Note: exact matching does not detect semantic overlap; review similar intents manually.")


if __name__ == "__main__":
    main()
