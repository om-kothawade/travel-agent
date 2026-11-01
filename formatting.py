"""Turn an Itinerary into Markdown (shown on the page and offered as a download)."""
from models import Itinerary


def _p(text: str) -> str:
    """Keep the model's line breaks when rendered as Markdown."""
    return text.strip().replace("\n", "  \n")


def to_markdown(it: Itinerary) -> str:
    icon = "✅" if it.good_time_to_visit else "⚠️"
    L = [
        f"# ✈️ Your trip to {it.destination}",
        "",
        _p(it.trip_overview),
        "",
        f"## {icon} Best time to visit",
        _p(it.time_verdict),
    ]
    if it.better_months:
        L += ["", f"**Better months:** {', '.join(it.better_months)}"]
    L += ["", "## 💰 Budget", _p(it.budget_breakdown)]

    for d in it.days:
        L += [
            "", "---", "",
            f"## 📍 Day {d.day}: {d.title}",
            f"*{d.date}*",
            "",
            f"**🌅 Morning**\n\n{_p(d.morning)}",
            "",
            f"**☀️ Afternoon**\n\n{_p(d.afternoon)}",
            "",
            f"**🌙 Evening**\n\n{_p(d.evening)}",
            "",
            f"**🍽️ Food**\n\n{_p(d.food_suggestions)}",
            "",
            f"**🚕 Getting around**\n\n{_p(d.getting_around)}",
            "",
            f"**💵 Estimated cost per person**\n\n{_p(d.estimated_cost_per_person)}",
            "",
            f"**🌤️ Weather:** {_p(d.weather_note)}",
            "",
            f"**💡 Tips:** {_p(d.tips)}",
        ]

    if it.packing_tips:
        L += ["", "---", "", "## 🎒 Packing tips"] + [f"- {t}" for t in it.packing_tips]
    if it.general_tips:
        L += ["", "## 🧭 General tips"] + [f"- {t}" for t in it.general_tips]
    L += ["", "Have a great trip! 🌍"]
    return "\n".join(L)
