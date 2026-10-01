import time
import re
from pathlib import Path

import chromadb
import ollama


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection(
    name="enterprise_knowledge"
)


# ============================================================
# STRIP MODEL REASONING
# ============================================================

def strip_thinking(text):

    # Case 1: properly paired <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Case 2: Ollama's qwen3 template opens <think> implicitly,
    # so the returned content often has only the closing tag
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    return text.strip()


# ============================================================
# RETRIEVE RELEVANT KNOWLEDGE
# ============================================================

def retrieve(query, number_of_results=3):

    start_time = time.time()

    try:
        embedding_response = ollama.embed(
            model="nomic-embed-text",
            input=query
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed — is Ollama running? ({e})")

    query_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieval_time = time.time() - start_time

    return documents, metadatas, retrieval_time


# ============================================================
# BUILD KNOWLEDGE CONTEXT
# ============================================================

def build_context(documents, metadatas):

    start_time = time.time()

    context_parts = []

    for document, metadata in zip(documents, metadatas):

        source = metadata.get(
            "source",
            "Unknown source"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{document}"
        )

    context = "\n\n".join(context_parts)

    context_time = time.time() - start_time

    return context, context_time


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, verbose=True):

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    documents, metadatas, retrieval_time = retrieve(
        question,
        number_of_results=3
    )

    if not documents:
        answer = "I don't have enough information in the knowledge base to answer that."
        if verbose:
            print("\nAnswer:\n")
            print(answer)
        return answer, []

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context, context_time = build_context(
        documents,
        metadatas
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an enterprise knowledge assistant. /no_think

Answer the user's question using ONLY the knowledge context provided below.

Rules:
- Give only the final answer.
- Do not show reasoning, analysis, or internal thoughts.
- Do not mention the knowledge context or how you found the answer.
- Do not invent information.
- If the answer is not available in the context, say:
  "I don't have enough information in the knowledge base to answer that."
- Keep the answer concise and relevant.

KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""

    # --------------------------------------------------------
    # LLM GENERATION (non-streaming, thinking disabled + stripped)
    # --------------------------------------------------------

    llm_start = time.time()

    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            think=False,
            stream=False,
            options={
                "temperature": 0.2,
                "num_predict": 700,
            }
        )
        raw_answer = response["message"]["content"]

    except TypeError:
        # Installed ollama package doesn't support 'think' kwarg
        response = ollama.chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False
        )
        raw_answer = response["message"]["content"]

    except Exception as e:
        llm_time = time.time() - llm_start
        error_answer = f"LLM generation failed — is Ollama running? ({e})"
        if verbose:
            print("\nAnswer:\n")
            print(error_answer)
        return error_answer, metadatas

    llm_time = time.time() - llm_start

    answer = strip_thinking(raw_answer)

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    for metadata in metadatas:

        source = metadata.get(
            "source",
            "Unknown source"
        )

        if source not in sources:
            sources.append(source)

    # --------------------------------------------------------
    # OUTPUT (only when run directly, not when imported)
    # --------------------------------------------------------

    if verbose:

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        for source in sources:
            print(f"- {source}")

        print()
        print(f"Retrieval time: {retrieval_time:.2f} seconds")
        print(f"Context building time: {context_time:.2f} seconds")
        print(f"LLM time: {llm_time:.2f} seconds")

    return answer, sources


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    question = input("Ask a question: ")

    generate_answer(question)