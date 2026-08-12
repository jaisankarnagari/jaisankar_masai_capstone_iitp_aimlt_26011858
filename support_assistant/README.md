Zepto Support Assistant

Overview
- This module embeds a short corpus of Zepto policy documents and exposes a simple FastAPI app to query them.
- All LLM calls default to a deterministic mock when `MOCK_LLM` is unset or set to `1`.

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
- The ingestion script uses `sentence-transformers/all-MiniLM-L6-v2` and ChromaDB to persist embeddings locally.
- If you set `MOCK_LLM=0`, the API returns a placeholder indicating real LLM integration isn't configured; extend `api.py` to call a provider when desired.
- This folder contains the `docs/` corpus used for embeddings.
