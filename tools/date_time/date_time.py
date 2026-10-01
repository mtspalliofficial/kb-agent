import re
from datetime import datetime, date, timedelta

TOOL_INFO = {
    "name": "date_time",
    "description": "Answers questions about today's date, days until/since a date, days between two dates, or a date N days from today."
}


def run(query):
    """
    Handles:
    - "today" / "current date"        -> today's date
    - "N days from/after today"       -> a future date
    - "days between DATE1 and DATE2"  -> days between two dates
    - "days until/since DATE"         -> days remaining/elapsed
    """

    if not query or not query.strip():
        return "Error: no query provided."

    query_lower = query.lower()

    # --------------------------------------------------------
    # "N days from/after today" — check this BEFORE the plain
    # "today" check, since both phrases contain the word "today"
    # --------------------------------------------------------
    offset_match = re.search(r"(\d+)\s*days?\s*(?:from|after)\s*(?:today|now)", query_lower)
    if offset_match:
        days = int(offset_match.group(1))
        target = date.today() + timedelta(days=days)
        return f"{target.strftime('%B %d, %Y')} ({days} days from today)."

    # --------------------------------------------------------
    # Plain "today" / "current date"
    # --------------------------------------------------------
    if "today" in query_lower or "current date" in query_lower:
        return date.today().strftime("%B %d, %Y")

    # --------------------------------------------------------
    # Find all YYYY-MM-DD dates in the query
    # --------------------------------------------------------
    all_dates = re.findall(r"(\d{4}-\d{2}-\d{2})", query)

    # --------------------------------------------------------
    # Two dates found -> "days between DATE1 and DATE2"
    # --------------------------------------------------------
    if len(all_dates) >= 2:
        try:
            date1 = datetime.strptime(all_dates[0], "%Y-%m-%d").date()
            date2 = datetime.strptime(all_dates[1], "%Y-%m-%d").date()
        except ValueError:
            return "Error: invalid date — use YYYY-MM-DD format."

        delta = abs((date2 - date1).days)
        return f"There are {delta} days between {date1} and {date2}."

    # --------------------------------------------------------
    # One date found -> "days until/since DATE"
    # --------------------------------------------------------
    if len(all_dates) == 1:
        try:
            target_date = datetime.strptime(all_dates[0], "%Y-%m-%d").date()
        except ValueError:
            return "Error: invalid date — use YYYY-MM-DD format."

        today = date.today()
        delta = (target_date - today).days

        if "since" in query_lower:
            return f"{abs(delta)} days have passed since {target_date}."

        if delta >= 0:
            return f"{delta} days remaining until {target_date}."
        else:
            return f"{abs(delta)} days have already passed since {target_date}."

    return "Error: could not find a valid date (use YYYY-MM-DD) or a recognized date query."


if __name__ == "__main__":
    print(run("what is today's date?"))
    print(run("how many days until 2026-12-31"))
    print(run("how many days since 2026-01-01"))
    print(run("how many days between 2026-10-15 and 2026-12-31"))
    print(run("what date is 90 days from today"))
    print(run("what date is 30 days from today"))
    print(run("2026-13-45"))
    print(run("what's the weather"))