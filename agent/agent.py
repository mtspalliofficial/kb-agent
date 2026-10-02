import sys
import re
import ast
from pathlib import Path

import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ask import generate_answer as knowledge_qa
from skills.policy_analysis import policy_analysis
from skills.knowledge_retrieval import knowledge_retrieval
from tools.calculator.calculator import run as run_calculator
from tools.date_time.date_time import run as run_date_time
from tools.document_summarization.document_summarization import run as run_summarizer

MODEL = "qwen3:4b"
VALID_INTENTS = {"knowledge_qa", "policy_analysis", "calculator", "date_time", "summarize", "out_of_scope"}


# ============================================================
# LLM HELPERS
# ============================================================

def _strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1].strip()


def _extract_marked(text, start="ANSWER_START", end="ANSWER_END"):
    if not text:
        return ""
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL | re.IGNORECASE)
    result = match.group(1).strip() if match else _strip_thinking(text)
    return result.replace(start, "").replace(end, "").strip()


def _call_llm(prompt, num_predict=300):
    try:
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                         think=False, stream=False,
                         options={"temperature": 0.0, "num_predict": num_predict})
        return r["message"]["content"]
    except TypeError:
        try:
            r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                             stream=False, options={"temperature": 0.0, "num_predict": num_predict})
            return r["message"]["content"]
        except Exception:
            return ""
    except Exception:
        return ""


# ============================================================
# FAST-PATH DETECTION
# ============================================================

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DAY = re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.IGNORECASE)
_DATE_WORDS = re.compile(
    r"\btoday\b|\bcurrent (?:date|year|time)\b|\bwhat time\b|\bday of the week\b|"
    r"\bthis year\b|\byesterday\b|\btomorrow\b|\bdays? ago\b|\bdays? (?:left|remaining)\b|"
    r"\bdays? between\b|\bdays? from (?:now|today)\b", re.IGNORECASE
)
_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|crore|crores|million|billion"
)
_CALC_WORDS = re.compile(
    r"\bhow much\b|\bhow many\b|\bcalculate\b|\bcompute\b|\bpercentage\b|\bpercent\b|%|"
    r"\bincreas\w*\b|\bdecreas\w*\b|\breduc\w*\b|\bdiscount\b|\btax\b|\bbudget\b|\bcost\b|"
    r"\bsavings?\b|\btotal\b|\bremain\w*\b|\bleft\b|\bbalance\b|\bdifference\b|\bmore\b|"
    r"\bless\b|\bfewer\b|\btimes\b|\bmultiplied\b|\bdivided\b|\bplus\b|\bminus\b|\badd\b",
    re.IGNORECASE
)


def _looks_like_date_question(q):
    return bool(_ISO_DATE.search(q) or _MONTH_DAY.search(q) or _DATE_WORDS.search(q))


def _looks_numerically_complex(q):
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", q)
    has_symbols = bool(re.search(r"[+\-*/×÷]", q))
    has_words = bool(re.search(rf"\b(?:{_NUMBER_WORDS})\b", q, re.IGNORECASE))
    has_calc = bool(_CALC_WORDS.search(q))
    if has_symbols and numbers:
        return True
    if len(numbers) >= 2:
        return True
    if has_words and has_calc:
        return True
    return False


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(question, debug=False):
    if _looks_like_date_question(question):
        if debug:
            print("[DEBUG] Fast-path: date_time")
        return "date_time"

    if _looks_numerically_complex(question):
        if debug:
            print("[DEBUG] Fast-path: calculator")
        return "calculator"

    prompt = f"""Classify the question into EXACTLY ONE category. /no_think

- knowledge_qa: general facts about enterprise AI — capabilities, risks, implementation approach, success metrics
- policy_analysis: specifically about an organization's policy, rules, governance requirements, retention periods, or compliance obligations
- summarize: summarize, condense, extract key points, give an overview
- out_of_scope: unrelated to enterprise AI, or not answerable from a knowledge base

(Math and date questions are handled separately and won't reach you here.)

Respond EXACTLY:
ANSWER_START
<category>
ANSWER_END

QUESTION: {question}"""

    extracted = _extract_marked(_call_llm(prompt, num_predict=150)).lower()
    if debug:
        print(f"[DEBUG] Classification: {extracted!r}")

    for intent in VALID_INTENTS:
        if extracted == intent:
            return intent

    q = question.lower()
    if any(w in q for w in ("summarize", "summarise", "summary", "key points", "overview")):
        return "summarize"
    if any(w in q for w in ("policy", "governance", "compliance", "responsible ai", "oversight", "requirement", "retain", "anonymiz")):
        return "policy_analysis"
    return "knowledge_qa"


# ============================================================
# MATH: TEXT NORMALIZATION
# ============================================================

FRACTION_WORDS = {
    "three quarters": 0.75, "two thirds": 2 / 3, "one third": 1 / 3, "a third": 1 / 3,
    "one quarter": 0.25, "a quarter": 0.25, "quarter": 0.25,
    "one half": 0.5, "a half": 0.5, "half": 0.5,
}

_WRAPPER_PHRASES = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute|find|determine)\s*:?\s*", re.IGNORECASE
)
_ALLOWED_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd}


def _normalize_math_text(text):
    text = text.translate(str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}))
    text = re.sub(r"[₹$€£]", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    for pattern, repl in [(r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"),
                           (r"\bdivided by\b", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-")]:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _validate_expression(expr):
    if not expr or len(expr) > 500 or not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    # Includes the operator node types themselves (ast.Add, ast.Sub, etc.)
    # in the allowed set — ast.walk() visits these as real nodes, and
    # omitting them here caused every valid expression to fail validation.
    allowed_types = (ast.Expression, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, *_ALLOWED_OPS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
        if isinstance(node, (ast.BinOp, ast.UnaryOp)) and type(node.op) not in _ALLOWED_OPS:
            return False
        if not isinstance(node, allowed_types):
            return False
    return True


def _wants_delta(q):
    return bool(re.search(r"how much.{0,20}(sav|reduc|increas|decreas)|amount (saved|reduced|increased|decreased)|(?:decreased|increased) by", q))


def _wants_new_value(q):
    return bool(re.search(r"new.{0,20}(cost|price|value|spending|amount)|final (?:price|amount|cost)", q))


def _pct_delta_or_new(base, percent, q, is_decrease):
    frac = float(percent) / 100
    if _wants_delta(q) and not _wants_new_value(q):
        return f"{base} * {frac}"
    return f"{base} * (1 - {frac})" if is_decrease else f"{base} * (1 + {frac})"


def _try_deterministic_patterns(cleaned, q):
    for phrase, value in sorted(FRACTION_WORDS.items(), key=lambda x: -len(x[0])):
        m = re.search(rf"\b{re.escape(phrase)}\s+of\s+(\d+(?:\.\d+)?)", q)
        if m:
            return f"{m.group(1)} * {value}"

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s+(?:[a-z\s']{0,40}?)?(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        return f"{m.group(2)} * ({float(m.group(1)) / 100})"

    m = re.search(r"increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*%?", cleaned, re.IGNORECASE)
    if m:
        return _pct_delta_or_new(m.group(1), m.group(2), q, is_decrease=False)

    m = re.search(r"decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*%?", cleaned, re.IGNORECASE)
    if m:
        return _pct_delta_or_new(m.group(1), m.group(2), q, is_decrease=True)

    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*increase\s*(?:on|of)?\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        return _pct_delta_or_new(m.group(2), m.group(1), q, is_decrease=False)

    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:reduction|decrease|discount|off)\s*(?:from|on|of|by)?\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        return _pct_delta_or_new(m.group(2), m.group(1), q, is_decrease=True)

    m = re.search(r"(?:increased|increases|went up|rose)\s*from\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        old, new = m.groups()
        return f"(({new} - {old}) / {old}) * 100" if "percent" in q else f"{new} - {old}"

    m = re.search(r"(?:decreased|decreases|fell|falls|dropped|drops|declined|went down)\s*from\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        old, new = m.groups()
        return f"(({old} - {new}) / {old}) * 100" if "percent" in q else f"{old} - {new}"

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|successful.{0,15}out of)\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m:
        return f"({m.group(1)} / {m.group(2)}) * 100"

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:per|every|each)\s*month", cleaned, re.IGNORECASE)
    if m:
        n_months = re.search(r"for\s*(\d+)\s*months", cleaned, re.IGNORECASE)
        if n_months:
            return f"{m.group(1)} * {n_months.group(1)}"
        if re.search(r"annual|year|annum", q):
            return f"{m.group(1)} * 12"

    return None


_MATH_FEWSHOT_PROMPT = """Convert the question into ONE arithmetic expression that computes the
FINAL requested answer. Translate meaning only — do not solve it yourself. /no_think

Rules:
- "X% of Y" = Y * (X/100)
- "remaining"/"left"/"balance" = value AFTER previous operations are applied
- ROI = ((gain - cost) / cost) * 100
- Use only: digits, decimal points, + - * / ( )
- Express percentages as decimal multiplication (15% = * 0.15), never as a % operator
- No words, currency symbols, variables, or commas
- If required information is missing, respond: INSUFFICIENT_INFO

Respond EXACTLY:
ANSWER_START
<expression OR INSUFFICIENT_INFO>
ANSWER_END

QUESTION: {question}"""


def extract_math_expression(question, allow_llm=True):
    if not question or not question.strip():
        return ""

    cleaned = _normalize_math_text(question)
    q_lower = cleaned.lower()

    stripped = _WRAPPER_PHRASES.sub("", cleaned).rstrip(" ?.")
    if _validate_expression(stripped):
        return stripped

    expr = _try_deterministic_patterns(cleaned, q_lower)
    if expr and _validate_expression(expr):
        return expr

    if not allow_llm:
        return ""

    raw = _call_llm(_MATH_FEWSHOT_PROMPT.format(question=cleaned), num_predict=200)
    extracted = _extract_marked(raw).strip()
    extracted = re.sub(r"```\w*", "", extracted).replace("```", "").strip()

    if not extracted or "INSUFFICIENT_INFO" in extracted.upper():
        return None

    return extracted if _validate_expression(extracted) else None


# ============================================================
# ROUTE HANDLERS
# ============================================================

def _handle_calculator(question):
    expr = extract_math_expression(question, allow_llm=True)
    if expr is None:
        return "I can't compute that because the question is missing information needed for the calculation, and I don't want to guess.", []
    if not expr:
        return "I couldn't identify a calculation in that question.", []
    try:
        return str(run_calculator(expr)), []
    except Exception as e:
        return f"Error evaluating expression: {e}", []


def _handle_date_time(question):
    try:
        return str(run_date_time(question)), []
    except Exception as e:
        return f"Error processing date query: {e}", []


def _handle_summarize(question):
    try:
        retrieved = knowledge_retrieval(question, number_of_results=6)
    except Exception as e:
        return f"Error retrieving document content: {e}", []
    if not retrieved:
        return "I don't have enough information in the knowledge base to summarize that.", []
    combined = "\n\n".join(item["content"] for item in retrieved)
    sources = list(dict.fromkeys(item["source"] for item in retrieved))
    try:
        return str(run_summarizer(query=question, source_text=combined)), sources
    except Exception as e:
        return f"Error generating summary: {e}", sources


def _handle_policy(question):
    try:
        result = policy_analysis(question)
        return result["answer"], result["sources"]
    except Exception as e:
        return f"Error performing policy analysis: {e}", []


def _handle_knowledge(question):
    try:
        return knowledge_qa(question, verbose=False)
    except Exception as e:
        return f"Error retrieving an answer: {e}", []


def _handle_out_of_scope(question):
    return "I don't have enough information in the knowledge base to answer that.", []


_HANDLERS = {
    "calculator": _handle_calculator, "date_time": _handle_date_time,
    "summarize": _handle_summarize, "policy_analysis": _handle_policy,
    "out_of_scope": _handle_out_of_scope, "knowledge_qa": _handle_knowledge,
}


def run_agent(question, verbose=True, debug=False):
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "intent": None}

    question = question.strip()
    intent = classify_intent(question, debug=debug)

    if verbose:
        print(f"[Agent] Selected route: {intent}")

    answer, sources = _HANDLERS[intent](question)
    return {"answer": answer, "sources": sources, "intent": intent}


if __name__ == "__main__":
    print("Enterprise AI Knowledge Assistant (type 'quit' to exit)\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        result = run_agent(question, verbose=True, debug=False)
        print("\nAnswer:\n")
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"- {s}")
        print(f"\n[Route taken: {result['intent']}]")
        print("-" * 60)