"""Turn an Itinerary into a PDF (fpdf2, pure Python, no system packages needed).

Unicode: the standard PDF fonts only cover Latin-1, so the rupee sign, curly
quotes, dashes etc. would break. We ship DejaVu Sans in ./fonts and use it.
Characters the font can't draw (e.g. Devanagari) are dropped instead of
crashing. If the fonts folder is missing we fall back to Helvetica with a
safe character mapping, so the PDF is still produced.
"""
import os
import re
import tempfile
import unicodedata

from fpdf import FPDF

from models import Itinerary

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
REGULAR_TTF = os.path.join(FONT_DIR, "DejaVuSans.ttf")
BOLD_TTF = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

ACCENT = (249, 115, 22)   # orange, same as the popup spinner
DARK = (34, 34, 34)
GREY = (110, 110, 110)

_ASCII_FALLBACK = {
    "₹": "Rs ", "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "•": "-", "…": "...", "\u00a0": " ", "\u202f": " ", "×": "x",
}


class _ItineraryPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 16, 16)
        self.unicode = os.path.exists(REGULAR_TTF) and os.path.exists(BOLD_TTF)
        if self.unicode:
            self.add_font("body", "", REGULAR_TTF)
            self.add_font("body", "B", BOLD_TTF)
            self.family = "body"
            from fontTools.ttLib import TTFont
            self._cmap = set(TTFont(REGULAR_TTF).getBestCmap().keys())
        else:
            self.family = "Helvetica"
            self._cmap = set(range(32, 256))

    # -- text safety ---------------------------------------------------------
    def clean(self, text: str) -> str:
        out = []
        for ch in str(text):
            if ch == "\n":
                out.append("\n")
            elif ch == "\t":
                out.append(" ")
            elif not self.unicode and ch in _ASCII_FALLBACK:
                out.append(_ASCII_FALLBACK[ch])
            elif ord(ch) in self._cmap and ord(ch) >= 32:
                out.append(ch)
            elif unicodedata.category(ch) == "Zs":
                out.append(" ")
            # anything else (unsupported glyph, control char) is dropped
        return re.sub(r"[ ]{2,}", " ", "".join(out)).strip()

    @property
    def bullet(self) -> str:
        return "\u2022" if self.unicode else "-"

    # -- page furniture ------------------------------------------------------
    def footer(self):
        self.set_y(-12)
        self.set_font(self.family, "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    # -- building blocks -----------------------------------------------------
    def title_text(self, text: str):
        self.set_font(self.family, "B", 20)
        self.set_text_color(*DARK)
        self.multi_cell(0, 9, self.clean(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(5)

    def heading(self, text: str, size: int = 14):
        if self.get_y() > self.h - 50:   # avoid a heading stranded at page bottom
            self.add_page()
        self.ln(3)
        self.set_font(self.family, "B", size)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 7, self.clean(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)
        self.ln(1)

    def day_banner(self, text: str):
        if self.get_y() > self.h - 60:
            self.add_page()
        self.ln(4)
        self.set_fill_color(*ACCENT)
        self.set_text_color(255, 255, 255)
        self.set_font(self.family, "B", 13)
        self.multi_cell(0, 9, " " + self.clean(text), fill=True,
                        new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)
        self.ln(2)

    def paragraph(self, text: str):
        self.set_font(self.family, "", 10.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.6, self.clean(text), align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def labelled(self, label: str, text: str):
        if self.get_y() > self.h - 35:
            self.add_page()
        self.set_font(self.family, "B", 10.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, self.clean(label), new_x="LMARGIN", new_y="NEXT")
        self.paragraph(text)

    def bullets(self, items: list[str]):
        self.set_font(self.family, "", 10.5)
        self.set_text_color(*DARK)
        for it in items:
            self.set_x(self.l_margin + 3)
            self.multi_cell(0, 5.6, f"{self.bullet} {self.clean(it)}", align="L",
                            new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def build_pdf(it: Itinerary) -> _ItineraryPDF:
    pdf = _ItineraryPDF()
    pdf.set_title(f"Trip to {it.destination}")
    pdf.add_page()

    pdf.title_text(f"Your trip to {it.destination}")
    pdf.paragraph(it.trip_overview)

    verdict = "Good time to visit" if it.good_time_to_visit else "Heads up: not the ideal time"
    pdf.heading(f"Best time to visit: {verdict}")
    pdf.paragraph(it.time_verdict)
    if it.better_months:
        pdf.labelled("Better months", ", ".join(it.better_months))

    pdf.heading("Budget")
    pdf.paragraph(it.budget_breakdown)

    for d in it.days:
        pdf.day_banner(f"Day {d.day}: {d.title}   ({d.date})")
        pdf.labelled("Morning", d.morning)
        pdf.labelled("Afternoon", d.afternoon)
        pdf.labelled("Evening", d.evening)
        pdf.labelled("Food", d.food_suggestions)
        pdf.labelled("Getting around", d.getting_around)
        pdf.labelled("Estimated cost per person", d.estimated_cost_per_person)
        pdf.labelled("Weather", d.weather_note)
        pdf.labelled("Tips", d.tips)

    if it.packing_tips:
        pdf.heading("Packing tips")
        pdf.bullets(it.packing_tips)
    if it.general_tips:
        pdf.heading("General tips")
        pdf.bullets(it.general_tips)

    pdf.ln(4)
    pdf.set_font(pdf.family, "", 10.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "Have a great trip!", align="C")
    return pdf


def save_pdf(it: Itinerary) -> str:
    """Write the PDF to a temp folder and return its path."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", it.destination).strip("_")[:40] or "trip"
    path = os.path.join(tempfile.mkdtemp(), f"itinerary_{safe}.pdf")
    build_pdf(it).output(path)
    return path