import sys
import re
from pathlib import Path

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ollama

from rag.ask import generate_answer as knowledge_qa
from skills.policy_analysis import policy_analysis
from skills.knowledge_retrieval import knowledge_retrieval

from tools.calculator.calculator import run as run_calculator
from tools.date_time.date_time import run as run_date_time
from tools.document_summarization.document_summarization import run as run_summarizer


# ============================================================
# SHARED HELPERS
# ============================================================

def _strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _extract_marked(text, start="ANSWER_START", end="ANSWER_END"):
    """
    Pulls out only the text between explicit markers we asked the model
    to emit. Falls back to <think> stripping if markers are missing.
    """
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return _strip_thinking(text)


def _call_llm(prompt, num_predict=300):

    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            stream=False,
            options={"temperature": 0.0, "num_predict": num_predict}
        )
        raw = response["message"]["content"]

    except TypeError:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw = response["message"]["content"]

    return raw


# ============================================================
# STEP 1 — INTENT CLASSIFICATION (SKILL SELECTION)
# ============================================================

VALID_INTENTS = [
    "knowledge_qa",
    "policy_analysis",
    "calculator",
    "date_time",
    "summarize",
    "out_of_scope"
]


def classify_intent(question, debug=False):
    """
    Uses the LLM to decide which skill/tool should handle the question.
    Only trusts the LLM's answer if it actually produced the ANSWER_START /
    ANSWER_END markers — never scans raw/leaked reasoning text for category
    names, since that text lists every category while "thinking out loud"
    and causes false matches on whichever one appears first.
    Falls back to keyword rules if markers are missing or unrecognized.
    """

    prompt = f"""
Classify the following user question into EXACTLY ONE of these categories. /no_think

Categories:
- knowledge_qa: a general factual question about enterprise AI answerable from a knowledge base
- policy_analysis: a question specifically about policy, rules, compliance, or governance needing structured analysis
- calculator: a question that requires a math calculation (percentages, totals, cost projections, ROI, break-even, etc.)
- date_time: a question about today's date, or days until/since/between specific dates
- summarize: a request to summarize, sum up, or give a brief overview of a document or topic (also covers "summarise")
- out_of_scope: unrelated to enterprise AI, or cannot be answered from a knowledge base

Respond in EXACTLY this format and nothing else:
ANSWER_START
<category name only>
ANSWER_END

QUESTION:
{question}
"""

    raw = _call_llm(prompt, num_predict=300)

    if debug:
        print("\n[DEBUG] Raw classification output:")
        print(raw)
        print("[DEBUG END]\n")

    match = re.search(r"ANSWER_START(.*?)ANSWER_END", raw, re.DOTALL)

    if match:
        category_text = match.group(1).strip().lower()
        for intent in VALID_INTENTS:
            if intent in category_text:
                return intent

    # --------------------------------------------------------
    # FALLBACK: keyword rules — used whenever markers are missing
    # or the marked content doesn't match a known category
    # --------------------------------------------------------

    question_lower = question.lower()

    if any(word in question_lower for word in [
        "calculate", "how much is", "%", "percent", "total cost", "sum of",
        "roi", "break-even", "break even", "savings", "annual", "per month",
        "per week", "per year", "reduction", "increase"
    ]):
        return "calculator"

    if any(word in question_lower for word in [
        "today's date", "what day is", "days until", "days since",
        "how many days", "days between", "date is"
    ]):
        return "date_time"

    if any(word in question_lower for word in [
        "summarize", "summarise", "summary", "tl;dr", "brief overview", "sum up",
        "key points"
    ]):
        return "summarize"

    if any(word in question_lower for word in [
        "policy", "regulation", "compliance", "governance", "requirement", "retain", "anonymiz"
    ]):
        return "policy_analysis"

    return "knowledge_qa"


# ============================================================
# STEP 2 — EXTRACT A MATH EXPRESSION (FOR CALCULATOR TOOL)
# ============================================================

FRACTION_WORDS = {
    "three quarters": 0.75,
    "two thirds": 2 / 3,
    "one third": 1 / 3,
    "one quarter": 0.25,
    "a quarter": 0.25,
    "quarter": 0.25,
    "one half": 0.5,
    "half": 0.5,
}


def _percent_delta_or_new(base, percent, question_lower, direction):
    """
    direction: 'increase' or 'decrease'
    Decides whether the question wants the NEW total, or just the
    AMOUNT changed (saved/increased), based on phrasing.
    """
    wants_new_value = re.search(r"new.{0,20}(cost|price|value|spending|amount)", question_lower)
    wants_delta = re.search(r"how much.{0,20}(sav|reduc|increas)|amount (saved|reduced|increased)", question_lower)

    percent_fraction = float(percent) / 100

    if direction == "decrease":
        if wants_new_value and not wants_delta:
            return f"{base} * (1 - {percent_fraction})"
        return f"{base} * {percent_fraction}"  # amount saved/reduced
    else:  # increase
        if wants_delta and not wants_new_value:
            return f"{base} * {percent_fraction}"  # amount increased
        return f"{base} * (1 + {percent_fraction})"  # new value


def extract_math_expression(question):
    """
    Converts a natural-language calculator question into a plain arithmetic
    expression (digits + - * / ( ) .).

    Returns:
        - a valid expression string if one can be determined
        - ""   if no calculation could be identified at all
        - None if the question is missing information needed to compute an
                answer (the caller should refuse rather than guess)
    """

    cleaned = question.replace(",", "")
    cleaned = re.sub(r"[₹$€£]", "", cleaned)
    q_lower = cleaned.lower()

    # --------------------------------------------------------
    # FRACTION WORDS: "three quarters of 2400"
    # --------------------------------------------------------
    for phrase, value in sorted(FRACTION_WORDS.items(), key=lambda x: -len(x[0])):
        m = re.search(rf"{re.escape(phrase)}\s+of\s+(\d+(?:\.\d+)?)", q_lower)
        if m:
            return f"{m.group(1)} * {value}"

    # --------------------------------------------------------
    # PERCENT OF: "15% of 2000000"
    # --------------------------------------------------------
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        percent, base = m.groups()
        return f"{base} * {float(percent) / 100}"

    # --------------------------------------------------------
    # PERCENTAGE INCREASE BETWEEN TWO VALUES: "increased from A to B"
    # --------------------------------------------------------
    m = re.search(r"(?:increased|went|rose)\s*from\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m and re.search(r"percent|percentage", q_lower):
        old, new = m.groups()
        return f"(({new} - {old}) / {old}) * 100"

    # --------------------------------------------------------
    # ROI: "costs A ... generates/saves B ... ROI"
    # --------------------------------------------------------
    if "roi" in q_lower or "return on investment" in q_lower:
        cost_m = re.search(r"costs?\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
        gain_m = re.search(r"(?:generates?|saves?|savings of)\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
        if cost_m and gain_m:
            cost, gain = cost_m.group(1), gain_m.group(1)
            return f"(({gain} - {cost}) / {cost}) * 100"
        return None  # ROI asked but missing cost or gain figure

    # --------------------------------------------------------
    # BREAK-EVEN: "costs A ... saves B per month ... months to break even"
    # --------------------------------------------------------
    if "break-even" in q_lower or "break even" in q_lower:
        cost_m = re.search(r"costs?\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
        monthly_m = re.search(r"(\d+(?:\.\d+)?)\s*per month", cleaned, re.IGNORECASE)
        if cost_m and monthly_m:
            return f"{cost_m.group(1)} / {monthly_m.group(1)}"
        return None

    # --------------------------------------------------------
    # REDUCTION / DECREASE (both phrasing orders)
    # --------------------------------------------------------
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:reduction|decrease|discount|off)\s*(?:from|on|of|by)?\s*(\d+(?:\.\d+)?)",
        cleaned, re.IGNORECASE
    )
    if m:
        percent, base = m.groups()
        return _percent_delta_or_new(base, percent, q_lower, "decrease")

    m = re.search(
        r"(\d+(?:\.\d+)?)[^%\d]{0,60}reduc\w*[^%\d]{0,30}by\s*(\d+(?:\.\d+)?)\s*%",
        cleaned, re.IGNORECASE
    )
    if m:
        base, percent = m.groups()
        return _percent_delta_or_new(base, percent, q_lower, "decrease")

    # --------------------------------------------------------
    # INCREASE (both phrasing orders)
    # --------------------------------------------------------
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*increase\s*(?:on|of)?\s*(\d+(?:\.\d+)?)",
        cleaned, re.IGNORECASE
    )
    if m:
        percent, base = m.groups()
        return _percent_delta_or_new(base, percent, q_lower, "increase")

    m = re.search(
        r"(\d+(?:\.\d+)?)[^%\d]{0,60}increas\w*[^%\d]{0,30}by\s*(\d+(?:\.\d+)?)\s*%",
        cleaned, re.IGNORECASE
    )
    if m:
        base, percent = m.groups()
        return _percent_delta_or_new(base, percent, q_lower, "increase")

    # --------------------------------------------------------
    # "X out of Y" as a percentage
    # --------------------------------------------------------
    m = re.search(r"(\d+(?:\.\d+)?)\s*out of\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m and re.search(r"percent|percentage|%", q_lower):
        part, whole = m.groups()
        return f"({part} / {whole}) * 100"

    # --------------------------------------------------------
    # MONTHLY -> ANNUAL
    # --------------------------------------------------------
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:per|every|each)\s*month", cleaned, re.IGNORECASE)
    if m and re.search(r"annual|year|annum", q_lower):
        return f"{m.group(1)} * 12"

    # --------------------------------------------------------
    # HOURS PER DAY x WORKING DAYS
    # --------------------------------------------------------
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*hours?\s*(?:every|per)\s*(?:working\s*)?day.*?(\d+(?:\.\d+)?)\s*(?:working\s*)?days",
        cleaned, re.IGNORECASE | re.DOTALL
    )
    if m:
        return f"{m.group(1)} * {m.group(2)}"

    # --------------------------------------------------------
    # EMPLOYEES x HOURS/WEEK x RATE x WEEKS
    # --------------------------------------------------------
    employees_m = re.search(r"(\d+(?:\.\d+)?)\s*employees", cleaned, re.IGNORECASE)
    hours_m = re.search(
        r"(\d+(?:\.\d+)?)\s*hours?\s*(?:(?:per|a|each|every)\s*week|weekly)",
        cleaned, re.IGNORECASE
    )
    rate_m = re.search(
        r"(?:(\d+(?:\.\d+)?)\s*(?:per hour|an hour|each hour|/hour|/hr))"
        r"|(?:hour\s*(?:is worth|costs?|valued at)\s*(\d+(?:\.\d+)?))",
        cleaned, re.IGNORECASE
    )
    weeks_m = re.search(r"(\d+(?:\.\d+)?)\s*weeks?\s*(?:this year|per year|a year)", cleaned, re.IGNORECASE)

    if hours_m and rate_m:
        rate = rate_m.group(1) or rate_m.group(2)
        weeks = weeks_m.group(1) if weeks_m else "52"

        if re.search(r"annual|year", q_lower) or weeks_m:
            if employees_m:
                return f"{employees_m.group(1)} * {hours_m.group(1)} * {rate} * {weeks}"
            return f"{hours_m.group(1)} * {rate} * {weeks}"

    if hours_m and not rate_m:
        # hours/week mentioned but no dollar rate given anywhere —
        # if the question wants a dollar figure, we can't compute it
        if re.search(r"\$|savings|cost|worth", q_lower):
            return None

    # --------------------------------------------------------
    # Plain expression already present
    # --------------------------------------------------------
    direct_match = re.search(r"[\d.\s+\-*/()]{3,}", cleaned)
    if direct_match:
        candidate = direct_match.group().strip()
        if any(ch.isdigit() for ch in candidate) and any(op in candidate for op in "+-*/"):
            return candidate

    # --------------------------------------------------------
    # LLM FALLBACK — for word-numbers and anything else unmatched
    # --------------------------------------------------------
    prompt = f"""
Convert the following question into a single arithmetic expression using
only digits, + - * / ( ) and decimal points. Convert any spelled-out
numbers (e.g. "twenty", "five thousand") into digits. /no_think

If the question is missing a number or value needed to compute an answer,
respond with exactly: INSUFFICIENT_INFO

Respond in EXACTLY this format and nothing else:
ANSWER_START
<expression OR INSUFFICIENT_INFO>
ANSWER_END

Examples:
Question: What is twenty percent of five thousand?
ANSWER_START
5000 * 0.20
ANSWER_END

Question: An AI system saves 25 hours per week. What are the annual savings?
ANSWER_START
INSUFFICIENT_INFO
ANSWER_END

QUESTION:
{cleaned}
"""

    raw = _call_llm(prompt, num_predict=200)
    extracted = _extract_marked(raw)

    if "INSUFFICIENT_INFO" in extracted.upper():
        return None

    allowed_chars = "0123456789+-*/(). "
    expression = "".join(c for c in extracted if c in allowed_chars).strip()

    if not any(ch.isdigit() for ch in expression):
        return None

    return expression


# ============================================================
# STEP 3 — AGENT ORCHESTRATION
# ============================================================

def run_agent(question, verbose=True, debug=False):
    """
    Main agent entry point.
    Returns a dict: {"answer": str, "sources": list, "intent": str}
    """

    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "intent": None}

    # --------------------------------------------------------
    # STEP 1: Understand request -> Select skill
    # --------------------------------------------------------

    intent = classify_intent(question, debug=debug)

    if verbose:
        print(f"[Agent] Selected route: {intent}")

    # --------------------------------------------------------
    # STEP 2: Dispatch to the correct skill or tool
    # --------------------------------------------------------

    if intent == "out_of_scope":
        return {
            "answer": "I don't have enough information in the knowledge base to answer that.",
            "sources": [],
            "intent": intent
        }

    if intent == "calculator":
        expression = extract_math_expression(question)

        if verbose:
            print(f"[Agent] Extracted expression: {expression}")

        if expression is None:
            return {
                "answer": "I can't compute that because the question is missing information needed for the calculation, and I don't want to guess at missing numbers.",
                "sources": [],
                "intent": intent
            }

        if not expression:
            return {
                "answer": "I couldn't identify a calculation in that question.",
                "sources": [],
                "intent": intent
            }

        result = run_calculator(expression)
        return {
            "answer": f"{result}",
            "sources": [],
            "intent": intent
        }

    if intent == "date_time":
        result = run_date_time(question)
        return {
            "answer": result,
            "sources": [],
            "intent": intent
        }

    if intent == "summarize":
        retrieved = knowledge_retrieval(question)

        if not retrieved:
            return {
                "answer": "I don't have enough information in the knowledge base to summarize that.",
                "sources": [],
                "intent": intent
            }

        combined_text = "\n\n".join(item["content"] for item in retrieved)
        sources = list(dict.fromkeys(item["source"] for item in retrieved))

        summary = run_summarizer(query="", source_text=combined_text)

        return {
            "answer": summary,
            "sources": sources,
            "intent": intent
        }

    if intent == "policy_analysis":
        result = policy_analysis(question)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent
        }

    # Default: knowledge_qa
    answer, sources = knowledge_qa(question, verbose=False)
    return {
        "answer": answer,
        "sources": sources,
        "intent": intent
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("Enterprise AI Knowledge Assistant (type 'quit' to exit)\n")

    while True:
        question = input("Ask a question: ")

        if question.strip().lower() in ("quit", "exit"):
            break

        result = run_agent(question, verbose=True, debug=False)

        print("\nAnswer:\n")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")

        print(f"\n[Route taken: {result['intent']}]")
        print("-" * 60)