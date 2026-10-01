import re
from pathlib import Path

import ollama

TOOL_INFO = {
    "name": "document_summarization",
    "description": "Summarizes a knowledge-base document by filename, or summarizes any block of text directly."
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"


# ============================================================
# STRIP MODEL REASONING (same fix as ask.py)
# ============================================================

def _strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


# ============================================================
# CORE: SUMMARIZE ANY TEXT
# ============================================================

def summarize_text(text, max_sentences=3):

    if not text or not text.strip():
        return "Error: no text provided to summarize."

    prompt = f"""
Summarize the following text in {max_sentences} sentences or fewer. /no_think

Rules:
- Give only the summary, nothing else.
- Do not show reasoning, analysis, or internal thoughts.
- Do not mention "the text" or "the document" — just summarize the content directly.
- Be concise and factual. Do not add information not present in the text.

TEXT:
{text}

SUMMARY:
"""

    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            stream=False,
            options={
                "temperature": 0.2,
                "num_predict": 500,
            }
        )
        raw_summary = response["message"]["content"]

    except TypeError:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw_summary = response["message"]["content"]

    except Exception as e:
        return f"Error: summarization failed — is Ollama running? ({e})"

    return _strip_thinking(raw_summary)


# ============================================================
# WHOLE-DOCUMENT MODE: SUMMARIZE A FILE BY NAME
# ============================================================

def summarize_document(filename, max_sentences=3):

    file_path = KNOWLEDGE_BASE_PATH / filename

    if not file_path.exists():
        return f"Error: document '{filename}' not found in knowledge base."

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return summarize_text(content, max_sentences=max_sentences)


# ============================================================
# TOOL ENTRY POINT (used by agent.py)
# ============================================================

def run(query, source_text=None):
    """
    If source_text is provided, summarizes that text directly.
    Otherwise, treats 'query' as a filename in the knowledge base.
    """

    if source_text:
        return summarize_text(source_text)

    return summarize_document(query)


if __name__ == "__main__":
    # Test 1: on-the-fly text summarization
    sample_text = """
    Enterprise AI adoption is accelerating across industries, but organizations
    face significant risks including data privacy concerns, security
    vulnerabilities, and issues with model accuracy and bias. Regulatory
    requirements are also becoming more stringent, requiring stronger human
    oversight and explainability in AI systems. Successful adoption depends on
    careful integration with existing systems and attention to operational
    reliability.
    """
    print("--- Text summarization ---")
    print(summarize_text(sample_text))

    # Test 2: whole-document summarization
    print("\n--- Document summarization ---")
    print(summarize_document("enterprise_ai_overview.txt"))