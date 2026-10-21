
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCE_FILE = DATA_DIR / "train.jsonl"
OUTPUT_FILE = DATA_DIR / "generated_review.jsonl"

examples = [
    {
        "user": "How do I publish a trip as a driver?",
        "assistant": (
            "Enter your starting point, destination, departure date and time, "
            "available seats, and passenger contribution amount. Review the "
            "details before publishing your trip."
        ),
    },
    {
        "user": "I want to travel from Kolenchery to Kochi.",
        "assistant": (
            "I can help you search for a ride from Kolenchery to Kochi. "
            "What date and approximate departure time would you prefer?"
        ),
    },
    {
        "user": "Are there any seats left in my selected ride?",
        "assistant": (
            "Seat availability must be checked against the latest booking "
            "information. Please select the ride so the app can check its "
            "current availability."
        ),
    },
    {
        "user": "Can you guarantee that my ride will be available tomorrow?",
        "assistant": (
            "I can't guarantee availability in advance. The app must check "
            "current ride listings, and availability may change before booking."
        ),
    },
    {
        "user": "The app found no rides for my route.",
        "assistant": (
            "Sorry, no matching rides were found. You could try a different "
            "departure time or a nearby pickup point if the app supports it, "
            "or check again later."
        ),
    },
    {
        "user": "Book the cheapest ride for me immediately.",
        "assistant": (
            "I'll need to search available rides and compare their displayed "
            "prices first. I'll show you a suitable option and ask you to "
            "confirm before submitting a booking."
        ),
    },
    {
        "user": "I need to cancel my ride.",
        "assistant": (
            "Open your booking details and check the cancellation option and "
            "applicable policy. The app must verify the booking before "
            "processing the cancellation."
        ),
    },
    {
        "user": "Can I change my departure time?",
        "assistant": (
            "That depends on the app's trip-editing rules and your booking "
            "status. Check the trip or booking details to see whether changing "
            "the departure time is supported."
        ),
    },
    {
        "user": "What if the driver has no available seats?",
        "assistant": (
            "You can't book a seat that isn't available. Search for another "
            "matching trip or try a different departure time."
        ),
    },
    {
        "user": "How much will my ride from Kolenchery to Kochi cost?",
        "assistant": (
            "The cost depends on the available trips and their displayed "
            "passenger contributions. I'll need current ride-search results "
            "to give you an accurate price."
        ),
    },
]


def load_existing_examples(path):
    results = []

    if not path.exists():
        return results

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return results


def to_training_example(item):
    return {
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]
    }


def main():
    existing = load_existing_examples(SOURCE_FILE)
    generated = [to_training_example(item) for item in examples]

    # Keep the original dataset unchanged.
    # Write only the new examples to a separate review file.
    existing_questions = {
        message["content"].strip().casefold()
        for item in existing
        for message in item.get("messages", [])
        if message.get("role") == "user"
    }

    unique_generated = []
    seen = set(existing_questions)

    for item in generated:
        question = item["messages"][0]["content"].strip().casefold()

        if question not in seen:
            unique_generated.append(item)
            seen.add(question)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for item in unique_generated:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Existing examples: {len(existing)}")
    print(f"New examples written: {len(unique_generated)}")
    print(f"Review file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
