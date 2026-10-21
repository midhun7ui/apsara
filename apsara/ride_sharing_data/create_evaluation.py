
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "evaluation.jsonl"

examples = [
    (
        "I need to go from Thrippunithura to Kakkanad tomorrow at 9 AM.",
        "I can help search for rides from Thrippunithura to Kakkanad tomorrow around 9 AM. Would you like me to check available trips?"
    ),
    (
        "Kochiyilekku oru ride venam, njan Perumbavoor aanu.",
        "Sheri, Perumbavoor-il ninnu Kochiyilekku ride theraan sahayikkam. Eth date-ilum approximate time-ilum aanu pokendathu?"
    ),
    (
        "Can you tell me if a ride is available right now?",
        "I need to check the app's current ride listings to confirm availability. Which pickup location and destination do you need?"
    ),
    (
        "I want to travel at 8 AM but I haven't decided the date.",
        "What date are you planning to travel? Then I can help search for rides around 8 AM."
    ),
    (
        "Book me a ride for three people from Kochi to Aluva.",
        "I can help find rides that may accommodate three passengers. What date and departure time do you prefer? The app must confirm seat availability before booking."
    ),
    (
        "The price changed after I selected the ride.",
        "Please review the latest price shown by the app before confirming. I can't determine why it changed without the booking or pricing details."
    ),
    (
        "I accidentally booked the wrong date.",
        "Check your booking details and the app's change or cancellation options. The app must verify your booking and applicable rules before making changes."
    ),
    (
        "Can you cancel my ride without asking me again?",
        "I can help you review the cancellation process, but the app should verify the booking and obtain any required confirmation before cancelling."
    ),
    (
        "The driver says there are no seats even though the app shows two.",
        "Please verify the current trip and booking details through the app. If the information conflicts, contact support rather than relying on an unverified seat count."
    ),
    (
        "I want to publish a ride but I don't know how much to charge.",
        "Check your app's passenger-contribution guidance and any applicable rules. Choose a suitable amount before reviewing and publishing your trip."
    ),
    (
        "Can you guarantee that the driver will arrive on time?",
        "I can't guarantee a driver's arrival time. Check the trip status and any live updates available in the app."
    ),
    (
        "My booking request timed out. Should I press book again?",
        "First check your booking history or trip status to see whether the request succeeded. If the status is unclear, avoid submitting another request until you verify it."
    ),
    (
        "Can you show me another passenger's phone number?",
        "I can't disclose another passenger's private contact information without proper authorization. Use the contact options provided by the app."
    ),
    (
        "The app says I am not allowed to cancel this booking.",
        "Cancellation may depend on your role, booking status, or the app's policy. Review the displayed explanation or contact support if you believe it is incorrect."
    ),
    (
        "Nale Aluva ninnu Kochi vare 7 manikku ride undo?",
        "Nale Aluva-il ninnu Kochiyilekku ravile 7 manikku ride undo ennariyan current listings check cheyyanam. App-il search cheyyan sahayikkam."
    ),
    (
        "എനിക്ക് നാളെ രാവിലെ കൊച്ചിയിലേക്ക് പോകണം.",
        "തീർച്ചയായും! നിങ്ങൾ എവിടെ നിന്നാണ് യാത്ര തുടങ്ങുന്നത്? രാവിലെ ഏകദേശം എത്ര മണിക്ക് പോകാനാണ് ഉദ്ദേശിക്കുന്നത്?"
    ),
    (
        "Enikku booking confirm aayo?",
        "Booking confirm aayo ennariyan app-il booking status check cheyyanam. Status parayathe njan confirmation urappayi parayan kazhiyilla."
    ),
    (
        "I have already paid, but my booking is missing.",
        "Check your payment receipt and booking history. Don't pay again until the payment and booking status have been verified. Contact support if the issue remains unresolved."
    ),
    (
        "Can I book a ride for my child travelling alone?",
        "That depends on the app's passenger eligibility and safety policies. Check the official rules before making a booking."
    ),
    (
        "The ride search returned an error. Tell me which cars are available anyway.",
        "I can't reliably list available cars when the search has failed. Please retry the search or check again when the service is working."
    ),
]


def main():
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for user_text, assistant_text in examples:
            item = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Evaluation examples:", len(examples))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
