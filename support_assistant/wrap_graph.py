#!/usr/bin/env python
"""
FastAPI wrapper for Zepto Assistant (Task 5)
Function-based version (no class definitions)

Run with:
    uvicorn app:app --reload
"""

from fastapi import FastAPI
import uvicorn
app = FastAPI()
# -----------------------------
# Core assistant function
# -----------------------------
def run_zepto_assistant(query: str, mock_mode: bool = True) -> dict:
    if mock_mode:
        # Retrieval triggered for policy/delivery/refund type queries
        if any(word in query.lower() for word in ["refund", "return", "delivery", "cancel", "membership"]):
            return {
                "answer": "Refunds are available within 7 days for damaged or defective items.",
                "sources": ["doc_03_chunk_0", "doc_05_chunk_1"],
                "confidence": 1.0,
            }
        # General questions → no retrieval
        else:
            return {
                "answer": "I can only answer questions about Zepto policies right now.",
                "sources": [],
                "confidence": 1.0,
            }
    else:
        # Placeholder for real LLM mode
        return {
            "answer": "Zepto allows returns within 7 days for damaged items.",
            "sources": ["doc_03_chunk_0"],
            "confidence": 0.95,
        }

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Zepto Assistant API")

@app.post("/ask")
def ask(request: dict):
    query = request.get("query", "")
    return run_zepto_assistant(query, mock_mode=True)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("wrap_graph:app", host="127.0.0.1", port=8000, reload=True)
