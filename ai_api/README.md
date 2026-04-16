![Pipeline Status](https://xxx.xxx.xxx.xxx/ics-projects/ai/i-nest_api/badges/main/pipeline.svg)
![Coverage](https://xxx.xxx.xxx.xxx/ics-projects/ai/i-nest_api/badges/main/coverage.svg)

# Document QA & Chat API
**Versione 1.4 - 31/03/2026**

API per l'upload di documenti, indicizzazione in Qdrant e interazione conversazionale basata su RAG (Retrieval-Augmented Generation), con autenticazione JWT e gestione multi-tenant.

---

## Sommario
- [Caratteristiche](#caratteristiche)
- [Panoramica veloce](#panoramica-veloce)
- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Database: SQLite vs PostgreSQL](#database-sqlite-vs-postgresql)
- [Avvio dell'API REST](#avvio-dellapi-rest)
- [Migrazione SQLite -> Postgres](#migrazione-sqlite---postgres)
- [Autenticazione JWT](#autenticazione-jwt)
- [Documentazione & Swagger](#documentazione--swagger)
- [Endpoints principali](#endpoints-principali)
- [Esecuzione con Docker](#esecuzione-con-docker)
- [Front-end statico](#front-end-statico)
- [Testing](#testing)
- [Contribuire](#contribuire)
- [Licenza](#licenza)

---

## Caratteristiche
- **Upload** di file `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.json`
- **Estrazione** testo e chunking (dimensione configurabile)
- **Embeddings** con `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Indicizzazione** in Qdrant
- **Chat RAG** via Ollama locale — risposta completa (`POST /chat`) o **streaming SSE** (`POST /chat/stream`)
- **Autenticazione** JWT delegata a Django (toggle `SKIP_AUTH`)
- **Multi-tenant**: namespace per `username` e `collection`
- **Cronologia chat**: salvataggio conversazioni e ripresa tramite `conversation_id`
- **Gestione conversazioni**: elenco, lettura messaggi e **cancellazione** (`DELETE /conversations/{id}`)
- **Storico upload** persistito nel database, consultabile via API
- **Elenco** e **cancellazione** di collection
- **Database duale**: SQLite (default, sviluppo) o PostgreSQL (produzione), selezionabile via `DB_ENGINE`
- **Configurazione** via `python-dotenv`

---

## Panoramica veloce
- Avvio rapido con Docker Compose: `docker compose up -d --build`
- Salute servizio: `curl http://localhost:9000/healthz`
- Documentazione swagger: `http://localhost:9000/docs`
- Dati persistenti: volume `./uploads` montato nel container API

---

## Prerequisiti
- Python ≥ 3.8
- Qdrant in esecuzione
- Ollama server in esecuzione
- PostgreSQL (opzionale, per produzione)
- `git`, `pip`

---

## Installazione
```bash
git clone <URL-repo>
cd i-nest_api
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.ps1    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configurazione
Crea un file `.env` nella root con:

```dotenv
# Directory per upload e temp
UPLOAD_DIR=./uploads

# Database (sqlite | postgres)
DB_ENGINE=sqlite
UPLOAD_DB_PATH=./uploads/uploads.sqlite3   # usato solo se DB_ENGINE=sqlite

# PostgreSQL (richiesto se DB_ENGINE=postgres)
PG_HOST=localhost
PG_PORT=5432
PG_DB=app_db
PG_USER=postgres
PG_PASSWORD=postgres
PG_SSLMODE=prefer

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_TIMEOUT=60

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
QDRANT_SCORE_THRESHOLD=0.3
ENABLE_RAG_DEBUG=false
ENABLE_RERANK=true
ENABLE_MMR=true
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
CHAT_RESULT_LIMIT=30
CHAT_CANDIDATES=30
CHAT_EXPANSION_LIMIT=3
CHAT_EXPANSION_CANDIDATES=12
CHAT_CONTEXT_CHAR_BUDGET=7000
CHAT_HISTORY_PROMPT_LIMIT=30

# Conversazioni
CONVERSATION_PREVIEW_CHARS=120
MAX_CONVERSATION_LIST=50

# Django JWT (in sviluppo SKIP_AUTH=true)
DJANGO_VERIFY_URL=http://django-server/api/token/verify/
SKIP_AUTH=true

# CORS (lista origini separate da virgola; usa * per accettare tutto in sviluppo)
CORS_ORIGINS=http://localhost:9001

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
LOG_MAX_BYTES=5000000
LOG_BACKUP_COUNT=5

# Upload
MAX_UPLOAD_SIZE_BYTES=20000000
```

### Descrizione variabili

| Variabile | Default | Descrizione |
|---|---|---|
| `DB_ENGINE` | `sqlite` | Backend database: `sqlite` o `postgres` |
| `UPLOAD_DB_PATH` | `./uploads/uploads.sqlite3` | Percorso SQLite (solo se `DB_ENGINE=sqlite`) |
| `PG_HOST/PORT/DB/USER/PASSWORD` | — | Connessione PostgreSQL (richiesti se `DB_ENGINE=postgres`) |
| `SKIP_AUTH` | `false` | `true` bypassa la verifica JWT (solo sviluppo) |
| `CHUNK_SIZE` | `500` | Dimensione chunk in caratteri |
| `CHUNK_OVERLAP` | `50` | Sovrapposizione tra chunk consecutivi |
| `QDRANT_SCORE_THRESHOLD` | `0.3` | Punteggio minimo risultati Qdrant |
| `ENABLE_RAG_DEBUG` | `false` | Log diagnostici aggiuntivi per chunking e retrieval |
| `ENABLE_RERANK` / `ENABLE_MMR` / `ENABLE_STITCH` | `true` | Controllo pipeline RAG avanzata (`ENABLE_RERANK` per il punteggio combinato, `ENABLE_MMR` per la diversificazione finale) |
| `ENABLE_MULTI_VECTOR_SEARCH` | `true` | Ricerca multi-vettore con espansione query |
| `CHAT_RESULT_LIMIT` / `CHAT_CANDIDATES` | `30` | Chunk recuperati e candidati interrogati per chat |
| `CHAT_EXPANSION_LIMIT` / `CHAT_EXPANSION_CANDIDATES` | `3` / `12` | Parametri espansione multi-vettore |
| `CHAT_CONTEXT_CHAR_BUDGET` | `7000` | Caratteri massimi passati al modello (~2000 token) |
| `CHAT_HISTORY_PROMPT_LIMIT` | `30` | Turni di cronologia inclusi nel prompt |
| `CONVERSATION_PREVIEW_CHARS` | `120` | Caratteri anteprima nell'elenco conversazioni |
| `MAX_CONVERSATION_LIST` | `50` | Limite massimo per `GET /conversations` |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Modello Ollama di default |
| `OLLAMA_TIMEOUT` | `60` | Timeout (secondi) richieste Ollama |
| `ALLOW_EMBEDDING_FALLBACK` | `false` | Abilita solo in test/debug il fallback deterministico degli embedding; se `false`, gli endpoint RAG rispondono `503` quando il modello non è disponibile |
| `CORS_ORIGINS` | `http://localhost:9001` | Origini CORS ammesse (`,`-separated; `*` per tutte) |
| `LOG_LEVEL` | `INFO` | Livello log |
| `LOG_FILE` | `./logs/app.log` | Percorso log con rotazione |
| `LOG_MAX_BYTES` | `5000000` | Dimensione massima per file di log prima della rotazione |
| `LOG_BACKUP_COUNT` | `5` | Numero di file di backup mantenuti |
| `MAX_UPLOAD_SIZE_BYTES` | `20000000` | Dimensione massima file upload (20 MB) |

---

## Database: SQLite vs PostgreSQL

L'applicazione supporta due backend tramite la variabile `DB_ENGINE`:

- **`sqlite`** (default): adatto per sviluppo locale. Il database viene creato automaticamente al percorso `UPLOAD_DB_PATH`.
- **`postgres`**: raccomandato per produzione. Richiede le variabili `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`. Lo schema viene creato automaticamente all'avvio.

In produzione con Docker Compose, il compose si collega alla rete esterna `postgres_project_default` e imposta `DB_ENGINE=postgres`.

---

## Avvio dell'API REST
Con **Uvicorn**:
```bash
uvicorn app:app --host 0.0.0.0 --port 9000 --workers 4
```

In background:
```bash
nohup uvicorn app:app --host 0.0.0.0 --port 9000 > uvicorn.log 2>&1 &
```

Oppure direttamente:
```bash
python app.py
```

---

## Migrazione SQLite -> Postgres
Lo script `migrate_sqlite_to_postgres.py` consente di migrare on-demand i dati da SQLite (tabelle `uploads`, `conversations`, `conversation_messages`) verso Postgres.

### Prerequisiti
- Database Postgres esistente e raggiungibile con utente/password e permessi di scrittura.
- File SQLite sorgente (default `./uploads/uploads.sqlite3`).
- Dipendenze installate (`SQLAlchemy`, `psycopg[binary]` presenti in `requirements.txt`).

### Esecuzione tipica
```bash
python migrate_sqlite_to_postgres.py \
  --sqlite-path ./uploads/uploads.sqlite3 \
  --pg-host localhost --pg-port 5432 \
  --pg-db mosaico --pg-user postgres --pg-password postgres \
  --batch-size 500
```

Flag utili:
- `--dry-run` legge e valida senza scrivere su Postgres.
- `--verbose` abilita l'echo SQL di SQLAlchemy.

Variabili d'ambiente equivalenti (CLI ha precedenza): `SQLITE_PATH`, `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`, `PG_SSLMODE`, `BATCH_SIZE` (default 500).

Comportamento:
- Converte timestamp ISO in `TIMESTAMPTZ` UTC.
- Upsert su `upload_id` e `conversation_id`; i messaggi sono deduplicati con chiave `(conversation_id, role, content, created_at)`.
- Crea automaticamente lo schema su Postgres; se su SQLite alcune tabelle non esistono, vengono saltate con log.

---

## Autenticazione JWT
Tutti gli endpoint protetti richiedono header:
```
Authorization: Bearer <JWT_TOKEN>
```
- Con `SKIP_AUTH=true`, qualsiasi token (anche fittizio) supera la verifica.
- In produzione (`SKIP_AUTH=false`), il token viene validato contro `DJANGO_VERIFY_URL`.

---

## Documentazione & Swagger
- **Swagger UI**: `http://<IP>:9000/docs`
- **ReDoc**:        `http://<IP>:9000/redoc`
- **OpenAPI JSON**: `http://<IP>:9000/openapi.json`

---

## Endpoints principali

### GET /
Verifica lo stato dell'app:
```json
{ "message": "I-NEST_API is running!" }
```

### POST /upload
Carica e indicizza un documento.
- Headers: `Authorization: Bearer <token>`
- Form data:
  - `file` (obbligatorio)
  - `username`, `collection` (opzionali)
  - Formati ammessi: `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.json`
  - Limite dimensione: `MAX_UPLOAD_SIZE_BYTES` (default 20 MB)

**Risposta**:
```json
{
  "message": "File caricato e processato con successo",
  "upload_id": "c6e86f08-...",
  "processing_info": {
    "upload_id": "c6e86f08-...",
    "collection_used": "user_coll",
    "num_chunks": 10,
    "num_points_inserted": 10,
    "original_filename": "contratto.pdf",
    "stored_filename": "c6e86f08-....pdf"
  }
}
```

### GET /uploads
Elenca la cronologia degli upload (filtrabile per utente o collection).
- Headers: `Authorization: Bearer <token>`
- Query params opzionali: `username`, `collection`, `status`, `upload_id`, `limit` (default 100, max 500)

**Risposta**:
```json
{
  "count": 2,
  "uploads": [
    {
      "upload_id": "c6e86f08-...",
      "username": "mario",
      "collection": "contratti",
      "original_filename": "contratto.pdf",
      "stored_filename": "c6e86f08-....pdf",
      "status": "completed",
      "num_chunks": 10,
      "created_at": "2024-07-12T10:15:00Z"
    }
  ]
}
```

### POST /chat
Interroga l'IA sul contenuto dei documenti indicizzati.
- Headers: `Authorization: Bearer <token>`
- Form data:
  - `question` (obbligatorio)
  - `username`, `collection` (opzionali)
  - `conversation_history` (opzionale): JSON lista turni `{role, content}` o testo libero
  - `conversation_id` (opzionale): continua una conversazione esistente; se assente ne crea una nuova
  - `model` (opzionale): override del modello Ollama per questa richiesta

**Risposta**:
```json
{
  "message": "Risposta generata dal modello...",
  "conversation_id": "b5742e9c-..."
}
```

### POST /chat/stream
Identico a `POST /chat` ma restituisce la risposta come **Server-Sent Events** (SSE): i token appaiono progressivamente nel browser senza attendere la risposta completa.

- Headers: `Authorization: Bearer <token>`, `Accept: text/event-stream`
- Form data: stessi campi di `POST /chat`

**Stream di eventi**:
```
data: {"chunk": "La "}
data: {"chunk": "risposta "}
data: {"chunk": "arriva..."}
data: {"done": true, "conversation_id": "b5742e9c-..."}
```

---

### GET /conversations
Restituisce le ultime conversazioni di un utente, ordinate per `updated_at` discendente.
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale), `limit` (default 10, max definito da `MAX_CONVERSATION_LIST`)

**Risposta**:
```json
{
  "count": 2,
  "conversations": [
    {
      "conversation_id": "b5742e9c-...",
      "title": "Domanda iniziale...",
      "message_count": 6,
      "last_message_preview": "Ultima risposta troncata...",
      "created_at": "2025-01-01T10:00:00Z",
      "updated_at": "2025-01-01T10:05:00Z",
      "collection": "documents"
    }
  ]
}
```

### GET /conversations/{conversation_id}
Restituisce i messaggi di una conversazione in ordine cronologico.
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale)

**Risposta**:
```json
{
  "conversation_id": "b5742e9c-...",
  "title": "Domanda iniziale...",
  "message_count": 4,
  "messages": [
    {"role": "user", "content": "Ciao", "created_at": "..."},
    {"role": "assistant", "content": "Salve!", "created_at": "..."}
  ],
  "created_at": "...",
  "updated_at": "...",
  "collection": "documents"
}
```

### DELETE /conversations/{conversation_id}
Elimina una conversazione e tutti i suoi messaggi.
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale)

**Risposta**:
```json
{ "deleted": "b5742e9c-..." }
```

Se la conversazione non esiste o non appartiene all'utente, restituisce `404`.

---

### GET /collection
Elenca gli elementi di una collection con payload e id.
- Headers: `Authorization: Bearer <token>`
- Query params: `username`, `collection`, `limit` (default 100)

**Risposta**:
```json
{ "collection": "user_coll", "count": 5, "items": [ {...}, {...} ] }
```

### GET /collections
Elenca tutte le collections presenti in Qdrant.
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale). Se valorizzato, filtra per nome esatto `username` o prefisso `username_`.

**Risposta**:
```json
{ "count": 2, "collections": ["user_a", "org_b_docs"] }
```

### DELETE /collection
Elimina una collection (e la sua configurazione, se presente).
- Headers: `Authorization: Bearer <token>`
- Query params: `username`, `collection`

**Risposta**:
```json
{ "message": "Collection 'user_coll' eliminata con successo." }
```

### GET /collection/config
Legge la configurazione (scope prompt) di una collection.
- Headers: `Authorization: Bearer <token>`
- Query params: `username`, `collection`

**Risposta**:
```json
{
  "collection": "user_coll",
  "config": {
    "collection": "user_coll",
    "scope_prompt": "Sei un esperto in Infrastrutture Critiche",
    "created_at": "2026-02-16T10:00:00Z",
    "updated_at": "2026-02-16T10:00:00Z"
  }
}
```

Se non esiste configurazione, `config` vale `null`.

### PUT /collection/config
Crea o aggiorna la configurazione di una collection.
- Headers: `Authorization: Bearer <token>`
- Body JSON: `username` (opzionale), `collection` (opzionale), `scope_prompt` (obbligatorio, non vuoto)

### DELETE /collection/config
Elimina la configurazione di una collection.
- Headers: `Authorization: Bearer <token>`
- Query params: `username`, `collection`

> La configurazione viene usata da `POST /chat` per specializzare il prompt con l'ambito di pertinenza della collection.

### GET /ollama/models
Elenca i modelli disponibili sull'istanza Ollama configurata.
- Headers: `Authorization: Bearer <token>`

**Risposta**:
```json
{
  "count": 3,
  "models": ["gemma3:1b", "gpt-oss:20b", "llama3:8b"],
  "source": "http://localhost:11434/api/tags"
}
```

### GET /healthz
Verifica lo stato di Qdrant, Ollama, database e storage. Restituisce anche la configurazione runtime.

**Risposta** (stato ok):
```json
{
  "status": "ok",
  "qdrant": { "ok": true, "latency_ms": 12 },
  "ollama": { "ok": true, "latency_ms": 45 },
  "database": { "ok": true },
  "storage": { "upload_dir": "./uploads", "writable": true },
  "config": { "ollama_model": "gpt-oss:20b", "chunk_size": 500, "..." : "..." }
}
```

---

## Esecuzione con Docker

### Docker Compose (configurazione standard)
Il `docker-compose.yml` avvia l'API e la collega a reti Docker esterne per Qdrant e PostgreSQL. Ollama viene raggiunto tramite `host.docker.internal` (processo sul host o container su altro stack).

```bash
# 1) Assicurarsi che i container di Qdrant e Postgres siano attivi sui loro stack
# 2) Copiare e personalizzare il file .env (DB_ENGINE, PG_*, OLLAMA_URL, ecc.)
# 3) Avviare l'API
docker compose up -d --build

# 4) Verificare
curl http://localhost:9000/healthz
```

**Reti esterne richieste** (nomi di default):
- `qdrant_project_default` — stack Qdrant
- `postgres_project_default` — stack PostgreSQL

Se i nomi differiscono, aggiornare le voci `networks.*.name` in `docker-compose.yml`.

**Variabili d'ambiente impostate dal compose**:
```
QDRANT_HOST=qdrant
OLLAMA_URL=http://host.docker.internal:11434/api/generate
DB_ENGINE=postgres
PG_HOST=postgres
```

Per usare un modello Ollama non ancora scaricato sul host:
```bash
ollama pull gemma3:1b
```

### Solo container API (Qdrant/Ollama su altri host)
```bash
docker build -t i-nest_api:latest .
docker run --rm -it \
  -p 9000:9000 \
  -e QDRANT_HOST=<ip_qdrant> \
  -e QDRANT_PORT=6333 \
  -e OLLAMA_URL=http://<ip_ollama>:11434/api/generate \
  -e SKIP_AUTH=true \
  -v $(pwd)/uploads:/app/uploads \
  i-nest_api:latest
```

---

## Front-end statico
Serve `index.html` su `/static`:
```
http://<IP>:9000/static/index.html
```

---

## Testing
Esegui la suite:
```bash
pytest -q
```
Filtra un set di test:
```bash
pytest -q tests/test_uploads.py -k upload
```

---

## Contribuire
1. Issue o PR
2. Branch `feature/x`, `bugfix/y`
3. Test con `pytest`

---

## Licenza
MIT License
