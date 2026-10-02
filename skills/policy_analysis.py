import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieve import search_knowledge_base
import ollama

MODEL = "qwen3:4b"
TOP_K = 4
MAX_TOKENS = 900
TEMPERATURE = 0.1

FALLBACK = "Not covered by the provided knowledge base."

PLACEHOLDER_PATTERNS = (
    "<text>", "(your answer)", "[your answer]",
    "replace this", "insert answer", "your response here",
    "[our answer]", "(our answer)"
)


def _strip_code_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_field_via_regex(text, field_name):
    pattern = rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        value = match.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        return value if value else None
    return None


def _parse_policy_json(raw_text):
    for candidate in (raw_text, _strip_code_fences(raw_text)):
        try:
            data = json.loads(candidate)
            return {
                "policy_information": str(data.get("policy_information", FALLBACK)),
                "analysis": str(data.get("analysis", FALLBACK)),
                "practical_implication": str(data.get("practical_implication", FALLBACK)),
            }
        except (TypeError, json.JSONDecodeError):
            continue

    recovered = {}
    for field in ("policy_information", "analysis", "practical_implication"):
        value = _extract_field_via_regex(raw_text, field)
        recovered[field] = value if value else FALLBACK
    return recovered


def _clean_field(value):
    value = (value or "").strip()
    if not value or any(p in value.lower() for p in PLACEHOLDER_PATTERNS):
        return FALLBACK
    return value


def policy_analysis(question):

    if not question or not question.strip():
        return {"answer": "Error: no question provided.", "sources": []}

    question = question.strip()

    try:
        results = search_knowledge_base(question, number_of_results=TOP_K)
    except Exception as e:
        return {"answer": f"Error: knowledge retrieval failed — {e}", "sources": []}

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []

    if not documents:
        return {
            "answer": (
                f"Policy information: {FALLBACK}\n\n"
                f"Analysis: {FALLBACK}\n\n"
                f"Practical implication: {FALLBACK}"
            ),
            "sources": []
        }

    context_parts = []
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        source = metadata.get("source", "Unknown source")
        context_parts.append(f"SOURCE: {source}\n{document}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an Enterprise AI Policy Assistant.

Answer the user's question using ONLY the supplied knowledge base.

RULES:
- Base every statement on the knowledge base provided below.
- You MAY combine, synthesize, and explain multiple related points from
  the knowledge base in your own words — this is expected for "why" and
  "how" questions — but never introduce a fact, rule, or number that is
  not present in the knowledge base.
- Do not make assumptions beyond what the knowledge base supports.
- If the knowledge base genuinely does not address the question's topic
  at all, use exactly: "{FALLBACK}"
- Keep each field to 2-4 sentences — thorough but not exhaustive.
- Return ONLY valid JSON. No markdown, no code fences, no text outside
  the JSON object.

Return exactly these three fields:

{{
  "policy_information": "...",
  "analysis": "...",
  "practical_implication": "..."
}}

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=False,
            format="json",
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
        )
    except TypeError:
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json",
                options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
            )
        except Exception as e:
            return {"answer": f"Error: policy analysis failed — {e}", "sources": []}
    except Exception as e:
        return {"answer": f"Error: policy analysis failed — is Ollama running? ({e})", "sources": []}

    raw_answer = response.get("message", {}).get("content", "").strip()
    if not raw_answer:
        return {"answer": "Error: the policy model returned an empty response.", "sources": []}

    data = _parse_policy_json(raw_answer)

    policy_information = _clean_field(data.get("policy_information"))
    analysis = _clean_field(data.get("analysis"))
    practical_implication = _clean_field(data.get("practical_implication"))

    final_answer = (
        f"Policy information: {policy_information}\n\n"
        f"Analysis: {analysis}\n\n"
        f"Practical implication: {practical_implication}"
    )

    sources = []
    for metadata in metadatas:
        metadata = metadata or {}
        source = metadata.get("source", "Unknown source")
        if source not in sources:
            sources.append(source)

    return {"answer": final_answer, "sources": sources}


if __name__ == "__main__":
    question = input("Enter a policy question: ").strip()
    result = policy_analysis(question)
    print("\nPolicy Analysis:\n")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")