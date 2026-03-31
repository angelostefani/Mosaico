# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mosaico** is a modular local Document QA & RAG (Retrieval-Augmented Generation) platform. It processes uploaded documents, generates embeddings, and answers questions via a local LLM — no cloud dependencies.

```
Browser → Django Frontend (:9001) → FastAPI AI API (:9000)
                                          ├→ Qdrant (:6333)   — Vector DB
                                          ├→ Ollama (:11434)  — Local LLM
                                          └→ PostgreSQL (:5433) — Persistence
```

## Running the Stack

```bash
# Start everything (standard)
docker compose up -d --build

# Include local Ollama container
docker compose --profile local up -d --build
```

Individual services also have their own `docker-compose.yml` files under `ai_api/`, `mosaico/`, `qdrant_project/`, `ollama_project/`, `postgres_project/`.

## Development (without Docker)

**AI API (FastAPI) — port 9000:**
```bash
cd ai_api
pip install -r requirements.txt
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 9000 --workers 4
```

**Frontend (Django) — port 9001:**
```bash
cd mosaico
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:9001
```

## Testing

```bash
cd ai_api
pytest -q
pytest -q tests/test_app.py -k upload   # Run a specific test
```

## Architecture

### ai_api/
- `app.py` — Monolithic FastAPI app (~2,200 lines): all endpoints, RAG pipeline, document ingestion, chat logic, Qdrant operations, Ollama calls.
- `db.py` — SQLAlchemy models and DB setup (supports SQLite for dev, PostgreSQL for production via `DB_ENGINE` env var).
- `tests/test_app.py` — Pytest test suite.
- `debug/` — Standalone debug scripts for upload, chat, and API.
- `migrate_sqlite_to_postgres.py` — One-shot migration tool.

**RAG flow:** document upload → text extraction (pdfplumber, python-docx, openpyxl) → chunking (default 500 chars, overlap 120) → embeddings (`sentence-transformers/all-MiniLM-L6-v2`) → Qdrant index → on chat, retrieve top-k chunks → Ollama LLM generates answer.

Multi-tenant isolation is achieved via Qdrant **collections** (one per namespace/user group).

### mosaico/ (Django frontend)
- `mosaico/settings.py` — Django config: CORS, JWT (60 min lifetime), logging, static files.
- `ui/views.py` — All page views: login, register, upload, chat, collection config, public chat.
- `ui/templates/` — Bootstrap HTML templates.
- JWT tokens are obtained from `/api/token/` and forwarded by Django views to the FastAPI backend on every request.

### Configuration (root `.env`)
Key variables:
- `OLLAMA_URL`, `OLLAMA_MODEL` — LLM connection
- `DB_ENGINE` (`sqlite` | `postgres`), `PG_*` — Database
- `QDRANT_HOST`, `QDRANT_PORT` — Vector DB
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL` — Ingestion tuning
- `CHAT_RESULT_LIMIT`, `CHAT_CANDIDATES`, `CHAT_CONTEXT_CHAR_BUDGET` — Retrieval tuning
- `ENABLE_RAG_DEBUG`, `ENABLE_RERANK`, `ENABLE_MMR`, `ENABLE_STITCH` — Feature flags
- `SKIP_AUTH` — Bypass JWT auth (development only)
- `DJANGO_SECRET_KEY`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS` — Django security
- `API_BASE`, `API_PUBLIC_BASE` — Django → FastAPI connection URLs

## Key API Endpoints (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload` | Ingest document |
| POST | `/chat` | RAG chat |
| GET | `/uploads` | Upload history |
| GET | `/conversations` | Conversation list |
| GET/PUT/DELETE | `/collection/config` | Collection settings |
| GET | `/healthz` | Full system health check |
| GET | `/docs` | Swagger UI |
