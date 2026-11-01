"""STEP 2: Custom function tools (weather) using Open-Meteo (free, no API key).

Key idea: the SDK turns each @function_tool's name, type hints and docstring
into the tool schema the model sees. Write docstrings for the MODEL.

We also summarise the data in Python instead of returning raw JSON. This
saves tokens and avoids the model mis-reading large numeric arrays.
"""
from datetime import date, datetime, timedelta

import requests
from agents import function_tool

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"


def _geocode(city: str):
    resp = requests.get(GEO_URL, params={"name": city, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not find a place called '{city}'.")
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city), r.get("country", "")


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _summarize(daily: dict) -> tuple[str, str]:
    """Return (per-day lines, overall summary) from an Open-Meteo `daily` block."""
    times = daily["time"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    rain = daily["precipitation_sum"]

    lines, rainy_days = [], 0
    for t, hi, lo, p in zip(times, tmax, tmin, rain):
        if hi is None or lo is None:
            continue
        p = p or 0
        if p >= 1:
            rainy_days += 1
        lines.append(f"{t}: {lo:.0f}-{hi:.0f}°C, rain {p:.1f}mm")

    valid_hi = [x for x in tmax if x is not None]
    valid_lo = [x for x in tmin if x is not None]
    if not valid_hi:
        return "", "No data."
    summary = (
        f"Avg high {sum(valid_hi)/len(valid_hi):.0f}°C, "
        f"avg low {sum(valid_lo)/len(valid_lo):.0f}°C, "
        f"{rainy_days}/{len(lines)} rainy days, "
        f"total rain {sum((x or 0) for x in rain):.0f}mm."
    )
    return "\n".join(lines), summary


@function_tool
def get_forecast(city: str, start_date: str, end_date: str) -> str:
    """Get the real day-by-day weather FORECAST for a city.
    Only works if the trip dates fall within the next 16 days from today.
    If it returns 'unavailable', use get_climate_history instead.

    Args:
        city: City name, e.g. "Jaipur".
        start_date: Trip start date, YYYY-MM-DD.
        end_date: Trip end date, YYYY-MM-DD.
    """
    try:
        today = date.today()
        start, end = _parse(start_date), _parse(end_date)
        last_ok = today + timedelta(days=15)
        if start > last_ok:
            return ("Forecast unavailable: trip starts more than 16 days away. "
                    "Use get_climate_history instead.")
        end = min(end, last_ok)

        lat, lon, name, country = _geocode(city)
        resp = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "daily": DAILY_VARS,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        }, timeout=15)
        resp.raise_for_status()
        lines, summary = _summarize(resp.json()["daily"])
        return f"FORECAST for {name}, {country}\n{lines}\nSUMMARY: {summary}"
    except Exception as e:
        return f"Error getting forecast: {e}"


@function_tool
def get_climate_history(city: str, start_date: str, end_date: str) -> str:
    """Get what the weather was like on the SAME calendar dates in each of the
    last 3 years. Use this when the trip is more than 16 days away, to judge
    whether it is a good time of year to visit (typical heat, cold, rain).

    Args:
        city: City name, e.g. "Jaipur".
        start_date: Trip start date, YYYY-MM-DD (the future trip date).
        end_date: Trip end date, YYYY-MM-DD.
    """
    try:
        today = date.today()
        start, end = _parse(start_date), _parse(end_date)
        lat, lon, name, country = _geocode(city)

        def shift(d: date, years: int) -> date:
            try:
                return d.replace(year=d.year - years)
            except ValueError:  # Feb 29
                return d.replace(year=d.year - years, day=28)

        # Find the first past year whose window has fully completed
        # (archive data lags by a few days).
        n = 1
        while shift(end, n) > today - timedelta(days=7):
            n += 1

        out = [f"CLIMATE HISTORY for {name}, {country} (same dates, past years)"]
        for k in range(n, n + 3):
            s, e = shift(start, k), shift(end, k)
            resp = requests.get(ARCHIVE_URL, params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "daily": DAILY_VARS,
                "start_date": s.isoformat(), "end_date": e.isoformat(),
            }, timeout=20)
            resp.raise_for_status()
            _, summary = _summarize(resp.json()["daily"])
            out.append(f"{s.year}: {summary}")
        return "\n".join(out)
    except Exception as e:
        return f"Error getting climate history: {e}"
