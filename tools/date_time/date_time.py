import re
from datetime import datetime, date, timedelta

TOOL_INFO = {
    "name": "date_time",
    "description": "Answers questions about today's date, days until/since a date (ISO or month-name), days between two dates, or a date N days from today."
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}
_MONTH_PATTERN = "|".join(_MONTHS.keys())


def _parse_month_day(text):
    match = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.IGNORECASE)
    if not match:
        return None
    month = _MONTHS[match.group(1).lower()]
    day = int(match.group(2))
    try:
        return date(date.today().year, month, day)
    except ValueError:
        return None


def run(query):

    if not query or not query.strip():
        return "Error: no query provided."

    query_lower = query.lower()
    today = date.today()

    offset_match = re.search(r"(\d+)\s*days?\s*(?:from|after)\s*(?:today|now)", query_lower)
    if offset_match:
        days = int(offset_match.group(1))
        target = today + timedelta(days=days)
        return f"{target.strftime('%B %d, %Y')} ({days} days from today)."

    if "today" in query_lower or "current date" in query_lower:
        return today.strftime("%B %d, %Y")

    all_dates = re.findall(r"(\d{4}-\d{2}-\d{2})", query)

    if len(all_dates) >= 2:
        try:
            date1 = datetime.strptime(all_dates[0], "%Y-%m-%d").date()
            date2 = datetime.strptime(all_dates[1], "%Y-%m-%d").date()
        except ValueError:
            return "Error: invalid date — use YYYY-MM-DD format."
        return f"There are {abs((date2 - date1).days)} days between {date1} and {date2}."

    if len(all_dates) == 1:
        try:
            target_date = datetime.strptime(all_dates[0], "%Y-%m-%d").date()
        except ValueError:
            return "Error: invalid date — use YYYY-MM-DD format."

        delta = (target_date - today).days
        if "since" in query_lower:
            return f"{abs(delta)} days have passed since {target_date}."
        if delta >= 0:
            return f"{delta} days remaining until {target_date}."
        return f"{abs(delta)} days have already passed since {target_date}."

    # Month-name date, e.g. "December 31" — handles the ISO patterns above
    # finding nothing, so this is the fallback for natural-language dates.
    month_day_matches = re.findall(rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", query, re.IGNORECASE)

    if len(month_day_matches) >= 2 and "between" in query_lower:
        try:
            d1 = date(today.year, _MONTHS[month_day_matches[0][0].lower()], int(month_day_matches[0][1]))
            d2 = date(today.year, _MONTHS[month_day_matches[1][0].lower()], int(month_day_matches[1][1]))
        except ValueError:
            return "Error: invalid date."
        return f"There are {abs((d2 - d1).days)} days between {d1.strftime('%B %d')} and {d2.strftime('%B %d')}."

    if len(month_day_matches) == 1:
        target = _parse_month_day(query)
        if target:
            if target < today and re.search(r"until|left|remaining", query_lower):
                target = date(today.year + 1, target.month, target.day)
            delta = (target - today).days
            if "since" in query_lower:
                return f"{abs(delta)} days have passed since {target.strftime('%B %d, %Y')}."
            if delta >= 0:
                return f"{delta} days remaining until {target.strftime('%B %d, %Y')}."
            return f"{abs(delta)} days have already passed since {target.strftime('%B %d, %Y')}."

    return "Error: could not find a valid date (use YYYY-MM-DD, a month name, or a recognized date query)."


if __name__ == "__main__":
    print(run("what is today's date?"))
    print(run("how many days until 2026-12-31"))
    print(run("how many days until December 31"))
    print(run("how many days since 2026-01-01"))
    print(run("how many days between 2026-10-15 and 2026-12-31"))
    print(run("what date is 90 days from today"))
    print(run("2026-13-45"))
    print(run("what's the weather"))