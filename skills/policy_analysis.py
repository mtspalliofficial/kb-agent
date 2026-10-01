import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieve import search_knowledge_base
import ollama


def _strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _extract_marked(text, start="ANSWER_START", end="ANSWER_END"):
    match = re.search(
        re.escape(start) + r"(.*?)" + re.escape(end),
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        result = match.group(1).strip()
    else:
        result = _strip_thinking(text)

    # Safety net: strip any leftover literal marker text
    result = result.replace(start, "").replace(end, "").strip()

    return result


def policy_analysis(question):

    if not question or not question.strip():
        return {"answer": "Error: no question provided.", "sources": []}

    results = search_knowledge_base(
        question,
        number_of_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {
            "answer": "I don't have enough information in the knowledge base to answer that.",
            "sources": []
        }

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"Source: {metadata.get('source', 'Unknown source')}\n"
            f"Content:\n{document}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a policy analysis assistant. /no_think

Analyze the user's question using ONLY the supplied knowledge base context.
Do not invent policies or facts.

Respond in EXACTLY this format and nothing else — no reasoning, no extra
text before ANSWER_START or after ANSWER_END. Replace each line with your
own real answer — never write the literal words "<text>" or leave a
placeholder. If the knowledge base does not contain relevant information,
write exactly "Not covered by the provided knowledge base." on all three
lines instead.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

ANSWER_START
Policy information: (your answer, or "Not covered by the provided knowledge base.")
Analysis: (your answer, or "Not covered by the provided knowledge base.")
Practical implication: (your answer, or "Not covered by the provided knowledge base.")
ANSWER_END
"""

    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            stream=False,
            options={"temperature": 0.2, "num_predict": 1400}
        )
        raw_answer = response["message"]["content"]
    except TypeError:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw_answer = response["message"]["content"]
    except Exception as e:
        return {
            "answer": f"Error: policy analysis failed — is Ollama running? ({e})",
            "sources": []
        }

    answer = _extract_marked(raw_answer)

    # Safety net: catch unfilled placeholders or empty output regardless
    # of whether the model followed the prompt's instructions
    if "<text>" in answer.lower() or not answer.strip():
        answer = "I don't have enough information in the knowledge base to answer that."

    return {
        "answer": answer,
        "sources": list(
            dict.fromkeys(
                metadata.get("source", "Unknown source")
                for metadata in metadatas
            )
        )
    }


if __name__ == "__main__":

    question = input("Enter a policy question: ")
    result = policy_analysis(question)

    print("\nPolicy Analysis:\n")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")