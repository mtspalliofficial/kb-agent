import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieve import search_knowledge_base


def knowledge_retrieval(query, number_of_results=3):

    if not query or not query.strip():
        return []

    start = time.time()

    try:
        results = search_knowledge_base(
            query,
            number_of_results=number_of_results
        )
    except Exception as e:
        print(f"Error: retrieval failed ({e})")
        return []

    retrieval_time = time.time() - start
    print(f"[Retrieval time: {retrieval_time:.2f}s]")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return []

    formatted_results = []

    for document, metadata in zip(documents, metadatas):
        formatted_results.append({
            "content": document,
            "source": metadata.get("source", "Unknown source")
        })

    return formatted_results


if __name__ == "__main__":

    question = input("Enter a question: ")
    results = knowledge_retrieval(question)

    if not results:
        print("\nNo relevant results found in the knowledge base.")
    else:
        print("\nKnowledge Retrieval Results:\n")
        for i, result in enumerate(results, start=1):
            print(f"--- Result {i} ---")
            print(f"Source: {result['source']}")
            print(result["content"])
            print()