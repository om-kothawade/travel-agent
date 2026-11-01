"""Test the agent alone (no UI).

Run:  python test_cli.py
Then open https://platform.openai.com/traces and inspect the runs.
Try a trip longer than 5 days (e.g. days=12) to see the chunked planning.
"""
import asyncio
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from agents import InputGuardrailTripwireTriggered  # noqa: E402

from formatting import to_markdown  # noqa: E402
from planning import TripRequest, plan_trip  # noqa: E402


async def main():
    start = date.today() + timedelta(days=5)   # change to +90 to test climate history
    req = TripRequest(
        destination="Jaipur",
        start=start,
        end=start + timedelta(days=2),         # try days=12 for a multi-chunk trip
        people=4,
        group_type="Family",
        budget_per_person="Rs 20,000 per person excluding flights",
        interests="history, street food, photography",
        extra_info="Flight lands in Jaipur at 2pm on day 1; departure flight at 8pm on the last day.",
    )
    try:
        itinerary = await plan_trip(req, on_progress=lambda f, m: print(f"[{f:.0%}] {m}"))
    except InputGuardrailTripwireTriggered:
        print("Blocked by guardrail: not a travel request.")
        return
    print(to_markdown(itinerary))


if __name__ == "__main__":
    asyncio.run(main())
