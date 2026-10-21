
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "generated_batch_2.jsonl"

examples = [
    ("I want to go to Kochi.", "Sure! Where will you be travelling from, and what date and approximate time would you prefer?"),
    ("I need a ride tomorrow.", "I can help you search for a ride. What are your pickup location, destination, and preferred departure time?"),
    ("Can I book two seats?", "I can help check rides that have at least two seats available. Which route and travel date do you need?"),
    ("I am travelling alone. Can I book?", "Yes, if a suitable ride has an available seat and meets the app's booking rules. Tell me your pickup, destination, and travel date so we can search."),
    ("I want to offer a ride to passengers.", "You can create a driver trip by entering your route, departure date and time, available seats, and passenger contribution. Review the details before publishing."),
    ("Can I publish a trip without a departure time?", "A departure time is normally needed to schedule a trip. Enter the intended date and time, or check the app's trip-creation requirements."),
    ("I entered the wrong destination.", "Please correct the destination before publishing or booking. If the trip or booking has already been submitted, check whether the app allows you to edit it."),
    ("Can I book a ride with zero available seats?", "No. A ride with zero available seats should not accept another passenger. Search for another trip or a different departure time."),
    ("The search is still loading.", "The ride search may be taking longer than expected. Wait briefly and try again. If the problem continues, the app should report a search error rather than inventing results."),
    ("The ride search failed.", "Sorry, I couldn't retrieve the ride results. Please try again. I won't claim that rides are available until the search succeeds."),
    ("The app shows no matching rides.", "No matching rides were returned for those search details. You could try another departure time or a nearby pickup point if supported."),
    ("Is the displayed price guaranteed?", "The displayed price is the information currently returned by the app. Check the final price and applicable terms before confirming your booking."),
    ("Can you tell me the exact number of available seats?", "I need current information for the selected trip to answer accurately. Please select a ride so the app can check its latest seat availability."),
    ("Book it without showing me the details.", "I can help with the booking, but you should review the selected ride, price, route, and departure time before confirming."),
    ("I changed my mind. Don't book it.", "No booking should be submitted unless you confirm it. If you haven't confirmed, we can stop here."),
    ("Can you cancel another passenger's booking?", "I can only help with actions authorized for your account. The app must verify your permissions before allowing a cancellation."),
    ("I cannot access my booking.", "Check that you're signed in to the correct account. If the booking still isn't visible, the app should help you troubleshoot or contact support."),
    ("Can I edit a trip after publishing it?", "That depends on the app's trip-editing rules and whether the trip has bookings. Check the trip details before making changes."),
    ("What if the driver cancels?", "Check the app for the latest booking status and any cancellation or refund guidance. The actual outcome depends on the app's policies."),
    ("Can I share my OTP with the driver?", "Don't share login or verification OTPs with a driver. Use only the app's official verification process."),
    ("എനിക്ക് കൊച്ചിയിലേക്ക് പോകണം.", "തീർച്ചയായും! നിങ്ങൾ എവിടെ നിന്നാണ് യാത്ര തുടങ്ങുന്നത്? യാത്രയുടെ തീയതിയും ഏകദേശ സമയവും പറയാമോ?"),
    ("Kolenchery ninnu Kochi vare ride venam.","Sheri, Kolenchery-il ninnu Kochiyilekku ride therayan sahayikkam. Eth date-il aanu pokendathu? Approximate time parayamo?"),
    ("Nale morning ride kittumo?", "ലഭ്യമായ rides ഉറപ്പാക്കാൻ app-ൽ ഇപ്പോഴത്തെ listings പരിശോധിക്കണം. ഏത് സ്ഥലത്ത് നിന്നാണ് യാത്ര തുടങ്ങേണ്ടത്, എവിടേക്കാണ് പോകേണ്ടത്?"),
    ("Enikku 3 seats venam.", "മൂന്ന് seats ലഭ്യമുള്ള ride കണ്ടെത്താൻ route-ഉം യാത്രാ തീയതിയും പറയൂ. നിലവിലെ seat availability app പരിശോധിക്കണം."),
    ("Can I pay the driver directly?", "Payment options depend on the app's actual payment policy. Check the payment instructions shown for your selected ride before paying."),
    ("The driver is late.", "Check the trip status and contact options provided in the app. If needed, contact support; I can't confirm the driver's location without live trip data."),
    ("Can you track the driver live?", "Live tracking is possible only if the app provides it and the relevant location data is available. Open your active trip to check."),
    ("I want a ride at 10 AM, but the date is not fixed.", "What date are you planning to travel? Once you provide the date, we can search for rides around 10 AM."),
    ("Can I reserve a seat for my friend?", "That depends on the app's booking rules. Check whether your account can book multiple seats and make sure the correct number of seats is selected."),
    ("What information do you need to search for a ride?", "Please provide your pickup location, destination, travel date, and preferred departure time. I can then help search for matching rides."),
    ("The booking button is not working.", "Please check your connection and try again once. If the issue continues, the app should report the booking error. Avoid submitting repeated requests if you aren't sure whether a booking succeeded."),
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

    print("Examples generated:", len(examples))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
