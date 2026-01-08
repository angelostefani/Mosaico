# Mosaico

Stack locale per Document QA e chat basata su RAG: API FastAPI con indicizzazione su Qdrant e generazione via Ollama, front-end Django e docker-compose separati per ogni servizio di contorno.

## Contenuto della repository
- `i-nest_api/` – API FastAPI (upload, indicizzazione, chat RAG, JWT, multi-tenant). Espone `:9000`. Vedi `i-nest_api/README.md`.
- `mosaico_django/` – Front-end Django + Bootstrap che consuma le API. Espone `:9001`. Vedi `mosaico_django/README.md`.
- `qdrant_project/` – Compose minimale per Qdrant (`:6333`, volume `qdrant_storage`).
- `ollama_project/` – Compose per Ollama (`:11434`, volume `ollama_data`, modello `gemma3:1b`).
- `postgres_project/` – Compose per Postgres (`:5433` -> `5432`, volume `postgres_data`).

## Prerequisiti generali
- Docker + Docker Compose
- Python 3.11+ se vuoi eseguire API o frontend fuori da Docker

## Avvio rapido (solo Docker)
1. Qdrant: `cd qdrant_project && docker compose up -d`
2. Ollama: `cd ollama_project && docker compose up -d && docker compose exec ollama ollama pull gemma3:1b`
3. Postgres (opzionale, se usi DB esterno a SQLite): `cd postgres_project && docker compose up -d`
4. API FastAPI: prepara `i-nest_api/.env` (variabili PG_* se usi Postgres) e avvia `cd i-nest_api && docker compose up -d`
5. Front-end Django: `cd mosaico_django && docker compose up -d` (profilo `frontend-dev` con `--profile dev` per sviluppo)

Le compose dei servizi di contorno creano le reti `qdrant_project_default`, `ollama_project_default` e `postgres_project_default` alle quali l'API (`i-nest_api/docker-compose.yml`) si aggancia.

## Porte e percorsi chiave
- API FastAPI: `http://localhost:9000` (serving anche `/static/index.html`)
- Front-end Django: `http://localhost:9001`
- Qdrant: `http://localhost:6333`
- Ollama: `http://localhost:11434`
- Postgres: `localhost:5433` (utente/password `postgres` di default)

## Documentazione dettagliata
Per istruzioni complete (installazione, endpoints, testing) consulta i README dei singoli moduli:
- `i-nest_api/README.md`
- `mosaico_django/README.md`
- `ollama_project/README.md`
- `qdrant_project/README.md`

