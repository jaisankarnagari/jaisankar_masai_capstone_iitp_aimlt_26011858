import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb

BASE_DIR = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_docs"

app = FastAPI(title="Zepto Support Assistant")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


def get_collection():
    # Use new PersistentClient API (replaces deprecated Settings-based client)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # collection may not exist
        raise RuntimeError("Chroma collection not found. Run ingestion first to populate docs.")
    return client, collection


def mock_llm_answer(question: str, contexts: List[str]):
    # Deterministic, rule-based mock: return the first context that contains any keyword from question
    q = question.lower()
    keywords = [w for w in q.replace('?', '').split() if len(w) > 3]
    for ctx in contexts:
        cl = ctx.lower()
        for kw in keywords:
            if kw in cl:
                return f"According to the docs: {ctx}"  # short, grounded answer
    # fallback: return the first context
    if contexts:
        return f"Based on available policy: {contexts[0]}"
    return "I don't have an answer for that."


@app.post('/query')
async def query(req: QueryRequest):
    try:
        client, collection = get_collection()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Retrieve top-k
    results = collection.query(query_texts=[req.question], n_results=req.top_k)
    # results structure: { 'ids': [[...]], 'metadatas': [[...]], 'documents': [[...]] }
    docs = []
    try:
        docs = results['documents'][0]
    except Exception:
        docs = []

    mock_mode = os.getenv('MOCK_LLM', '1') != '0'
    if mock_mode:
        answer = mock_llm_answer(req.question, docs)
        return {"answer": answer, "sources": [m.get('source') for m in results.get('metadatas',[[]])[0]], "llm_used": "mock"}

    # If MOCK_LLM=0, placeholder for real LLM integration (not configured)
    return {"answer": "Real LLM integration not configured in this environment.", "sources": [m.get('source') for m in results.get('metadatas',[[]])[0]], "llm_used": "none"}
