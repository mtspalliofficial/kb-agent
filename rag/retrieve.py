from pathlib import Path

import chromadb
import ollama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


# Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_collection(
    name="enterprise_knowledge"
)


def search_knowledge_base(query, number_of_results=3):
    """
    Search the knowledge base for information relevant to the query.
    """

    # Convert the user's question into an embedding
    response = ollama.embed(
        model="nomic-embed-text",
        input=query
    )

    query_embedding = response["embeddings"][0]

    # Search ChromaDB for the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    return results


if __name__ == "__main__":

    question = input("Ask a question: ")

    results = search_knowledge_base(question)

    print("\nRelevant knowledge:\n")

    for i, document in enumerate(results["documents"][0]):

        source = results["metadatas"][0][i]["source"]

        print(f"--- Result {i + 1} ---")
        print(f"Source: {source}")
        print(document)
        print()
