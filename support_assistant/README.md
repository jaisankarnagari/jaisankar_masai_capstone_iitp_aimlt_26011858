Zepto Support Assistant

Overview
- Embeds a short corpus of Zepto policy documents and exposes a FastAPI app to query them.

- Implements a Retrieval‑Augmented Generation (RAG) pipeline with intent classification, retrieval, and structured answer generation.

- All LLM calls default to a deterministic mock mode when MOCK_LLM is unset or set to 1.

- Real LLM mode (MOCK_LLM=0) is optional and validates outputs against a strict Pydantic schema.

- This module embeds a short corpus of Zepto policy documents and exposes a simple FastAPI app to query them.

- All LLM calls default to a deterministic mock when `MOCK_LLM` is unset or set to `1`.

- Real LLM mode (MOCK_LLM=0) is optional and validates outputs against a strict Pydantic schema.

Quick start

1. Install dependencies (no virtualenv):

```bash
python -m pip install -r requirements.txt --user
```

2. Ingest the documents (creates `support_assistant/chroma_db`):

```bash
python support_assistant/zepto_support_assistant.py
```

3. Run the API locally:

```bash
uvicorn support_assistant.api:app --host 127.0.0.1 --port 8000
```

4. Example query (mocked):

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"question":"What is the delivery fee?", "top_k":2}'
```

Notes
- The ingestion script uses `sentence-transformers/all-MiniLM-L6-v2` and ChromaDB (v0.4+) to persist embeddings locally.
- The code has been updated to use ChromaDB's new `PersistentClient` API (replacing deprecated Settings-based configuration).
- If you set `MOCK_LLM=0`, the API returns a placeholder indicating real LLM integration isn't configured; extend `api.py` to call a provider when desired.
- This folder contains the `docs/` corpus used for embeddings.
- ChromaDB automatically persists data with `PersistentClient`, so no explicit persist() call is needed.

Architecture
Pipeline Stages
1. Ingestion

    - File: zepto_support_assistant.py

    - Function: ingest_documents() loads .txt files from docs/, chunks them, and embeds with sentence-transformers/all-MiniLM-L6-v2.

    - Storage: Vectors persisted in ChromaDB (zepto_docs collection).

2. Embedding

    - Component: HuggingFace embedding function inside ChromaDB.

    - Output: Dense vector embeddings stored with metadata.

3. Retrieval

    - File: langgraph_flow.py

    - Node: retrieve_and_answer queries ChromaDB (real mode) or returns deterministic chunks (mock mode).

    - Output: Top‑k chunks + IDs passed forward.

4. Generation

    - Node: direct_answer (general questions) or retrieve_and_answer (policy questions).

    - Uses prompting.py templates in real mode.

    - Output validated against ZeptoAssistantOutput schema (answer, sources, confidence).

MOCK_LLM Toggle

    Mock Mode (MOCK_LLM=1)

    - Deterministic answers, fixed confidence, sources populated for policy queries.

    - No real LLM calls.

Real Mode (MOCK_LLM=0)

    - Calls LLM (placeholder), validates output with retries (up to 3).

    - Confidence varies (e.g., 0.9).

    - Corrective instructions applied if schema validation fails.

Data Flow Diagram

[Docs in /support_assistant/docs]
       |
       v
[Chunking + Embedding] --> [ChromaDB Collection]
       |
       v
[LangGraph Retrieval Node] --> [Generate Answer Node]
       |
       v
[Validated JSON Output: answer, sources, confidence]


Testing

Test Files
1.  test_langgraph_flow.py → Comprehensive workflow tests (intent classification, retrieval, generation).

2. test_support_assistant.py → Quick smoke tests.

3. test_diagnostic.py → Dependency and import diagnostics.

4. test_schema_validation.py → Pydantic schema validation tests.

5. test_wrap_graph.py → FastAPI endpoint tests.

6. test.docker.py → Docker build/run instructions.

Run Tests:
    - pytest support_assistant/test_langgraph_flow.py -v
    - python support_assistant/test_schema_validation.py
    - python support_assistant/test_diagnostic.py

Task 4: Pydantic Output Schema Validation
    class ZeptoAssistantOutput(BaseModel):
        answer: str
        sources: list[str]
        confidence: float = Field(..., ge=0.0, le=1.0)
- Mock Mode: Deterministic schema population, fixed confidence.

- Real Mode: Validates LLM output with up to 3 retries, applies corrective fixes.

- Error Handling: Returns error response with confidence=0.0 if validation fails.

- Test Coverage: Policy vs general queries, JSON round‑trip, constraint enforcement.

Task 5: FastAPI Wrapper
    - File: wrap_graph.py
    - Endpoint: POST /ask
    - Input: {"query": "What is your refund policy?"}

    Output:
        {
            "answer": "Refunds are available within 7 days for damaged or defective items.",
            "sources": ["doc_03_chunk_0", "doc_05_chunk_1"],
            "confidence": 1.0
        }
Task 6: Dockerization
        FROM python:3.11-slim
        WORKDIR /app
        COPY .. /app
        RUN pip install --no-cache-dir -r requirements.txt
        EXPOSE 8000
        CMD ["uvicorn", "support_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]

        Build & Run:
        docker build -t zepto-assistant .
        docker run -p 7860:7860 zepto-assistant

    Test with curl:
        curl -X POST "http://127.0.0.1:7860/ask" \
        -H "Content-Type: application/json" \
        -d '{"query":"What is your refund policy?"}'

Additional Notes
- Embedding model: sentence-transformers/all-MiniLM-L6-v2.

- Vector DB: ChromaDB v0.4+ with PersistentClient.

- Lazy loading optimizations in langgraph_flow.py for faster imports.

- Negative constraints enforced in prompting.py to avoid speculative answers.

- testing_guide.md provides troubleshooting steps for imports and performance.

Summary
- This project delivers a modular RAG pipeline with:

    - Document ingestion + embedding in ChromaDB

    - Intent classification and retrieval via LangGraph

    - Structured answer generation validated by Pydantic

    - FastAPI wrapper for serving queries

    - Dockerized deployment

    - Comprehensive test suites and guides