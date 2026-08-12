import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(directory: str):
    documents = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                documents.append((filename, content))
    return documents


def create_chromadb_collection():
    # Use new PersistentClient API (replaces deprecated Settings-based client)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    if COLLECTION_NAME in [col.name for col in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    hf = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=hf,
    )
    return client, collection


def ingest_documents():
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise FileNotFoundError(f"No text documents found in {DOCS_DIR}")

    client, collection = create_chromadb_collection()

    ids = []
    metadatas = []
    texts = []
    for filename, content in documents:
        ids.append(filename)
        metadatas.append({"source": filename})
        texts.append(content)

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas,
    )
    # Note: PersistentClient automatically persists data, no need for explicit persist() call
    print(f"Ingested {len(ids)} documents into collection '{COLLECTION_NAME}'")
    print(f"Chroma database persisted at: {PERSIST_DIR}")


if __name__ == "__main__":
    ingest_documents()
