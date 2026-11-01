"""Structured output schema. The planner agent must return an `Itinerary`."""
from pydantic import BaseModel


class Day(BaseModel):
    day: int
    date: str                       # YYYY-MM-DD
    title: str
    morning: str                    # detailed: places, timings, how to get there, costs
    afternoon: str
    evening: str
    food_suggestions: str           # named places/dishes for breakfast, lunch, dinner
    getting_around: str             # transport between stops + approx cost
    estimated_cost_per_person: str  # itemised daily estimate
    weather_note: str
    tips: str                       # bookings, timings, safety, etiquette


class Itinerary(BaseModel):
    destination: str
    trip_overview: str              # vibe of the trip, where to stay, how to pace it
    good_time_to_visit: bool
    time_verdict: str               # 1-3 sentences: is this a good time and why
    better_months: list[str]        # empty list if the chosen time is already good
    budget_breakdown: str           # per-person and group totals vs the stated budget
    days: list[Day]
    packing_tips: list[str]
    general_tips: list[str]         # money, SIM/connectivity, safety, customs, etc.
