"""Gradio UI. The itinerary is shown on the page and can be downloaded as a PDF.
No email, no external delivery.

Run locally:  python app.py
"""
import asyncio
import os
import time
from datetime import date, datetime
from html import escape

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # locally reads .env; on hosts, env vars are already set

from agents import InputGuardrailTripwireTriggered, MaxTurnsExceeded  # noqa: E402

from formatting import to_markdown  # noqa: E402
from pdf_export import save_pdf  # noqa: E402
from planning import CHUNK_DAYS, TripRequest, plan_trip  # noqa: E402

# No day limit by default. To protect your OpenAI bill on a public site, set e.g. 30.
MAX_TRIP_DAYS: int | None = None
MAX_PEOPLE = 50
COOLDOWN_SECONDS = 120
_last_request: dict[str, float] = {}  # simple in-memory rate limit per visitor

# ------------------------------------------------------------ centered popup
CSS = """
#overlay { padding: 0 !important; border: none !important; background: transparent !important; }
.ov-backdrop {
  position: fixed; inset: 0; z-index: 99999;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.55); backdrop-filter: blur(2px);
}
.ov-card {
  width: min(420px, 88vw); padding: 28px 24px; text-align: center;
  border-radius: 16px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
  background: var(--background-fill-primary, #fff);
  color: var(--body-text-color, #222);
}
.ov-card h3 { margin: 14px 0 6px; font-size: 1.2rem; }
.ov-card p  { margin: 4px 0; opacity: 0.9; }
.ov-card small { opacity: 0.6; }
.ov-spin {
  width: 48px; height: 48px; margin: 0 auto; border-radius: 50%;
  border: 5px solid rgba(128, 128, 128, 0.25); border-top-color: #f97316;
  animation: ov-rot 0.9s linear infinite;
}
@keyframes ov-rot { to { transform: rotate(360deg); } }
"""


def overlay(msg: str, started: float) -> str:
    elapsed = int(time.time() - started)
    return (
        f"<style>{CSS}</style>"
        '<div class="ov-backdrop"><div class="ov-card">'
        '<div class="ov-spin"></div>'
        "<h3>✈️ Planning your trip…</h3>"
        f"<p>{escape(msg)}</p>"
        f"<small>{elapsed}s elapsed · please keep this tab open</small>"
        "</div></div>"
    )


# ------------------------------------------------------------------- helpers
def _client_id(request: gr.Request | None) -> str:
    if request is None:
        return "local"
    fwd = request.headers.get("x-forwarded-for")  # set by hosts' proxies
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _parse_dates(start: str, end: str):
    try:
        s = datetime.strptime(start.strip(), "%Y-%m-%d").date()
        e = datetime.strptime(end.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format, e.g. 2026-12-20.")
    if s < date.today():
        raise ValueError("Start date can't be in the past.")
    if e < s:
        raise ValueError("End date must be on or after the start date.")
    if MAX_TRIP_DAYS and (e - s).days + 1 > MAX_TRIP_DAYS:
        raise ValueError(f"Please keep trips to {MAX_TRIP_DAYS} days or fewer.")
    return s, e


def _out(status="", md="", file_path=None, popup=""):
    """One tuple for all outputs: (status, itinerary, download file, popup)."""
    file_update = (gr.update(value=file_path, visible=True) if file_path
                   else gr.update(value=None, visible=False))
    return status, md, file_update, popup


# ------------------------------------------------------------------ the app
async def plan(destination, start, end, people, group_type, budget, interests,
               extra_info, request: gr.Request = None):
    """Generator: every yield updates (status, itinerary, download, popup)."""
    try:
        if not destination.strip():
            yield _out("❌ Please enter a destination.")
            return
        s, e = _parse_dates(start, end)
        try:
            n_people = int(people)
        except (TypeError, ValueError):
            raise ValueError("Please enter the number of people.")
        if not 1 <= n_people <= MAX_PEOPLE:
            raise ValueError(f"Number of people must be between 1 and {MAX_PEOPLE}.")

        visitor = _client_id(request)
        if time.time() - _last_request.get(visitor, 0) < COOLDOWN_SECONDS:
            yield _out("⏳ Please wait a couple of minutes before planning another trip.")
            return
        _last_request[visitor] = time.time()

        req = TripRequest(
            destination=destination.strip(),
            start=s,
            end=e,
            people=n_people,
            group_type=group_type,
            budget_per_person=budget.strip() or "flexible / mid-range",
            interests=interests.strip() or "general sightseeing",
            extra_info=extra_info or "",
        )

        started = time.time()
        state = {"msg": "Researching and checking weather…"}
        yield _out(popup=overlay(state["msg"], started))

        # Run the planner in the background; poll its progress to refresh the popup.
        task = asyncio.create_task(
            plan_trip(req, on_progress=lambda f, m: state.update(msg=m))
        )
        while not task.done():
            await asyncio.sleep(1)
            yield _out(popup=overlay(state["msg"], started))
        itinerary = await task  # re-raises any error from the planner

        md = to_markdown(itinerary)
        try:
            pdf_path = save_pdf(itinerary)
            yield _out("✅ Your itinerary is ready! You can also download it as a PDF below.",
                       md, pdf_path)
        except Exception as pdf_err:  # still show the itinerary if PDF creation fails
            yield _out(f"✅ Your itinerary is ready, but the PDF couldn't be created ({pdf_err}).", md)

    except ValueError as ve:
        yield _out(f"❌ {ve}")
    except InputGuardrailTripwireTriggered:
        yield _out("❌ I can only help with travel planning requests.")
    except MaxTurnsExceeded:
        yield _out("❌ The agent took too many steps. Try a simpler request.")
    except Exception as ex:
        yield _out(f"❌ Something went wrong: {ex}")


with gr.Blocks(title="AI Travel Planner") as demo:
    gr.Markdown(
        "# ✈️ AI Travel Itinerary Agent\n"
        "Researches the web, checks weather and best time to visit, and builds a "
        "detailed day-by-day itinerary right here on this page.\n\n"
        f"Longer trips are planned {CHUNK_DAYS} days at a time, so they take a few minutes."
    )
    with gr.Row():
        destination = gr.Textbox(label="Destination", placeholder="Jaipur")
    with gr.Row():
        start = gr.Textbox(label="Start date (YYYY-MM-DD)")
        end = gr.Textbox(label="End date (YYYY-MM-DD)")
    with gr.Row():
        people = gr.Number(label="Number of people", value=2, precision=0, minimum=1, maximum=MAX_PEOPLE)
        group_type = gr.Radio(["Family", "Friends"], value="Family", label="Travelling as")
    with gr.Row():
        budget = gr.Textbox(label="Budget per person", placeholder="₹20,000 per person (excluding flights)")
        interests = gr.Textbox(label="Interests", placeholder="history, street food, photography")
    extra_info = gr.Textbox(
        label="Additional info (optional)",
        placeholder="e.g. Flights: Bengaluru → Jaipur, lands 2pm on day 1, departs 8pm on last day. "
                    "Staying near Hawa Mahal. One vegetarian, a child aged 6.",
        lines=3,
        max_lines=6,
    )
    btn = gr.Button("Plan my trip", variant="primary")
    status = gr.Markdown()
    itinerary_md = gr.Markdown()
    download = gr.File(label="Download itinerary (PDF)", visible=False)
    popup = gr.HTML(elem_id="overlay")  # fixed-position popup; empty when idle

    btn.click(
        plan,
        [destination, start, end, people, group_type, budget, interests, extra_info],
        [status, itinerary_md, download, popup],
        show_progress="hidden",  # hide Gradio's default low-on-screen progress bar
    )

if __name__ == "__main__":
    # Hosts like Render give you a port in $PORT and need 0.0.0.0. Locally this
    # falls back to 127.0.0.1:7860 as before.
    on_cloud = "PORT" in os.environ
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0" if on_cloud else None,
        server_port=int(os.getenv("PORT", "7860")),
    )