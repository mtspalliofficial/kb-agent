import time
import json
import re
from pathlib import Path

import chromadb
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name="enterprise_knowledge")

FALLBACK = "I don't have enough information in the knowledge base to answer that."


def _strip_code_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_answer_via_regex(text):
    match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if match:
        value = match.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        return value if value else None
    return None


def _parse_answer_json(raw_text):
    for candidate in (raw_text, _strip_code_fences(raw_text)):
        try:
            data = json.loads(candidate)
            answer = str(data.get("answer", "")).strip()
            if answer:
                return answer
        except (TypeError, json.JSONDecodeError):
            continue
    return _extract_answer_via_regex(raw_text)


def retrieve(query, number_of_results=3):
    start_time = time.time()
    try:
        embedding_response = ollama.embed(model="nomic-embed-text", input=query)
    except Exception as e:
        raise RuntimeError(f"Embedding failed — is Ollama running? ({e})")

    query_embedding = embedding_response["embeddings"][0]
    results = collection.query(query_embeddings=[query_embedding], n_results=number_of_results)

    return results["documents"][0], results["metadatas"][0], time.time() - start_time


def build_context(documents, metadatas):
    start_time = time.time()
    context_parts = []
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown source")
        context_parts.append(f"Source: {source}\nContent:\n{document}")
    return "\n\n".join(context_parts), time.time() - start_time


def generate_answer(question, verbose=True):

    documents, metadatas, retrieval_time = retrieve(question, number_of_results=3)

    if not documents:
        if verbose:
            print("\nAnswer:\n")
            print(FALLBACK)
        return FALLBACK, []

    context, context_time = build_context(documents, metadatas)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the knowledge context provided below.
Do not invent information. If the answer is not available in the context,
the answer field must be exactly:
"{FALLBACK}"

KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Return ONLY valid JSON with one field:
{{"answer": "..."}}
"""

    llm_start = time.time()

    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            stream=False,
            format="json",
            options={"temperature": 0.2, "num_predict": 700}
        )
        raw_answer = response["message"]["content"]
    except TypeError:
        try:
            response = ollama.chat(
                model="qwen3:4b",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json",
                options={"temperature": 0.2, "num_predict": 700}
            )
            raw_answer = response["message"]["content"]
        except Exception as e:
            return f"Error: generation failed ({e})", []
    except Exception as e:
        error_answer = f"LLM generation failed — is Ollama running? ({e})"
        if verbose:
            print("\nAnswer:\n")
            print(error_answer)
        return error_answer, []

    llm_time = time.time() - llm_start
    answer = _parse_answer_json(raw_answer) or FALLBACK

    sources = []
    for metadata in metadatas:
        source = metadata.get("source", "Unknown source")
        if source not in sources:
            sources.append(source)

    if verbose:
        print("\nAnswer:\n")
        print(answer)
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
        print(f"\nRetrieval time: {retrieval_time:.2f}s")
        print(f"Context building time: {context_time:.2f}s")
        print(f"LLM time: {llm_time:.2f}s")

    return answer, sources


if __name__ == "__main__":
    question = input("Ask a question: ")
    generate_answer(question)