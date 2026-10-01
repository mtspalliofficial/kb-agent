from pathlib import Path

import chromadb
import ollama


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


# -----------------------------
# Find all knowledge documents
# -----------------------------

document_paths = sorted(KNOWLEDGE_BASE.glob("*.txt"))

if not document_paths:
    raise RuntimeError("No .txt files found in knowledge_base.")


# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="enterprise_knowledge"
)


# -----------------------------
# Rebuild the collection
# -----------------------------

existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])


# -----------------------------
# Process every document
# -----------------------------

total_chunks = 0

for document_path in document_paths:

    text = document_path.read_text(encoding="utf-8")

    sections = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if current_chunk:
            current_chunk += "\n\n" + section
        else:
            current_chunk = section

        if len(current_chunk) >= 500:
            chunks.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    print(f"Loaded document: {document_path.name}")
    print(f"Created {len(chunks)} meaningful chunks")

    # -----------------------------
    # Create embeddings
    # -----------------------------

    for index, chunk in enumerate(chunks):

        embedding_response = ollama.embed(
            model="nomic-embed-text",
            input=chunk
        )

        embedding = embedding_response["embeddings"][0]

        collection.upsert(
            ids=[f"{document_path.stem}_chunk_{index}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": document_path.name,
                    "chunk": index
                }
            ]
        )

        total_chunks += 1


# -----------------------------
# Finished
# -----------------------------

print()
print(f"Stored {total_chunks} embeddings in ChromaDB.")
print("RAG index rebuilt successfully.")