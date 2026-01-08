![Pipeline Status](https://10.196.1.1/ics-projects/ai/i-nest_api/badges/main/pipeline.svg)
![Coverage](https://10.196.1.1/ics-projects/ai/i-nest_api/badges/main/coverage.svg)

# Document QA & Chat API  
**Versione 1.2 - 25/11/2025**

API per l’upload di documenti, indicizzazione in Qdrant e interazione conversazionale basata su RAG (Retrieval-Augmented Generation), con autenticazione JWT e gestione multi-tenant.

---

## Sommario
- [Caratteristiche](#caratteristiche)  
- [Panoramica veloce](#panoramica-veloce)  
- [Prerequisiti](#prerequisiti)  
- [Installazione](#installazione)  
- [Configurazione](#configurazione)  
- [Avvio dell'API REST](#avvio-dellapi-rest)  
- [Migrazione SQLite -> Postgres](#migrazione-sqlite---postgres)  
- [Autenticazione JWT](#autenticazione-jwt)  
- [Documentazione & Swagger](#documentazione--swagger)  
- [Endpoints principali](#endpoints-principali)  
- [Front-end statico](#front-end-statico)  
- [Testing](#testing)  
- [Contribuire](#contribuire)  
- [Licenza](#licenza)  

---

## Caratteristiche
- **Upload** di file `.txt`, `.pdf`, `.doc`, `.docx`  
- **Estrazione** testo e chunking (dimensione configurabile)  
- **Embeddings** con `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **Indicizzazione** in Qdrant  
- **Chat RAG** via Ollama locale  
- **Autenticazione** JWT delegata a Django (toggle `SKIP_AUTH`)  
- **Multi-tenant**: namespace per `username` e `collection`  
- **Cronologia chat**: salvataggio conversazioni e ripresa delle ultime 10 per utente  
- **Storico upload** persistito in SQLite, consultabile via API  
- **Elenco** e **cancellazione** di collection  
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
- Redis (per rate-limiter, se abilitato)  
- `git`, `pip`  

---

## Installazione
\`\`\`bash
git clone <URL-repo>
cd i-nest_api
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.ps1    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

---

## Configurazione
Crea un file \`.env\` nella root con:

\`\`\`dotenv
# Directory per upload e temp
UPLOAD_DIR=./uploads
UPLOAD_DB_PATH=./uploads/uploads.sqlite3

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_TIMEOUT=60

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=0
QDRANT_SCORE_THRESHOLD=0.0
ENABLE_RAG_DEBUG=false
ENABLE_RERANK=true
CHAT_RESULT_LIMIT=3

# Django JWT (in sviluppo SKIP_AUTH=true)
DJANGO_VERIFY_URL=http://django-server/api/token/verify/
SKIP_AUTH=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
LOG_MAX_BYTES=5000000
LOG_BACKUP_COUNT=5
\`\`\`

- **`SKIP_AUTH=true`** bypassa la verifica JWT in sviluppo.  
- In produzione, impostalo a `false`.
- `UPLOAD_DB_PATH` consente di spostare il database degli upload (default sotto `uploads/`).
- `CHUNK_OVERLAP` definisce la sovrapposizione tra chunk consecutivi (default `0`).
- `QDRANT_SCORE_THRESHOLD` imposta il punteggio minimo dei risultati di ricerca Qdrant (default `0.0`).
- `ENABLE_RAG_DEBUG` abilita log diagnostici aggiuntivi per monitorare chunking e retrieval.
- `ENABLE_RERANK` consente di riordinare i risultati usando il reranker ibrido (default `true`).
- `CHAT_RESULT_LIMIT` definisce quanti chunk al massimo vengono recuperati per ogni domanda (default `3`).
- Ollama: `OLLAMA_URL` e `OLLAMA_TIMEOUT` (timeout in secondi, default 60) per la chiamata al modello.
- Logging: `LOG_LEVEL` (default `INFO`), `LOG_FILE` percorso del log con rotazione, `LOG_MAX_BYTES` dimensione massima per file di log prima della rotazione (default 5MB), `LOG_BACKUP_COUNT` numero di file di backup mantenuti.


---

## Avvio dell’API REST
Con **Uvicorn**:
\`\`\`bash
uvicorn app:app   --host 0.0.0.0   --port 9000   --workers 4
\`\`\`

Per mantenerlo in background:
\`\`\`bash
nohup uvicorn app:app --host 0.0.0.0 --port 9000 > uvicorn.log 2>&1 &
\`\`\`

---

## Migrazione SQLite -> Postgres
Lo script `migrate_sqlite_to_postgres.py` consente di migrare on-demand i dati da SQLite (tabelle `uploads`, `conversations`, `conversation_messages`) verso Postgres.

### Prerequisiti
- Database Postgres esistente e raggiungibile (es. istanza Docker su porta 5433) con utente/password e permessi di scrittura.
- File SQLite sorgente (default `./uploads/uploads.sqlite3`).
- Dipendenze installate (`SQLAlchemy`, `psycopg[binary]` presenti in `requirements.txt`).

### Esecuzione tipica
```bash
python migrate_sqlite_to_postgres.py --sqlite-path ./uploads/uploads.sqlite3 --pg-host localhost --pg-port 5433 --pg-db mosaico --pg-user postgres --pg-password postgres --batch-size 500

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

## Esecuzione con Docker

### Opzione A: Docker Compose (consigliata)
Esegue API, Qdrant e Ollama in un’unica rete:

```bash
# 1) (Facoltativo) impostare variabili in .env
# 2) Avvio dei servizi
docker compose up -d --build

# 3) Verifica stato API
curl http://localhost:9000/healthz
```

Note importanti:
- L’API parla con Qdrant e Ollama via i service name `qdrant` e `ollama`.
- In sviluppo, `SKIP_AUTH=true` è abilitato; in produzione impostalo a `false` e configura `DJANGO_VERIFY_URL`.
- La cartella `uploads/` è montata come volume su `/app/uploads` nel container API.
- Per usare un modello con Ollama, ad esempio `gemma3:1b`:

```bash
docker exec -it ollama ollama pull gemma3:1b
```

### Opzione B: Solo container API
Se Qdrant e Ollama girano su altri host/stack, costruisci e lancia solo l’API:

```bash
docker build -t i-nest_api:latest .
docker run --rm -it \
  -p 9000:9000 \
  -e QDRANT_HOST=<ip_o_hostname_qdrant> \
  -e QDRANT_PORT=6333 \
  -e OLLAMA_URL=http://<ip_o_hostname_ollama>:11434/api/generate \
  -e SKIP_AUTH=true \
  -v $(pwd)/uploads:/app/uploads \
  i-nest_api:latest
```

### Opzione C: API con Qdrant/Ollama avviati da compose separati
Se hai già attivi i progetti:
- `.../ai/ollama_project/docker-compose.yml`
- `.../ai/qdrant_project/docker-compose.yml`

allora questo repository espone un `docker-compose.yml` che avvia SOLO l’API e la collega alle reti esterne di quei progetti.

Passi:
1) Avvia prima i due stack esterni:
```bash
cd <path>/ai/ollama_project && docker compose up -d
cd <path>/ai/qdrant_project && docker compose up -d
```
2) Verifica i nomi delle reti (di default: `ollama_project_default` e `qdrant_project_default`):
```bash
docker network ls | grep -E "(ollama_project_default|qdrant_project_default)"
```
3) Se i nomi differiscono, aggiorna le voci `networks.*.name` in questo `docker-compose.yml` dell’API.
4) Avvia l’API:
```bash
docker compose up -d --build
```

Per la risoluzione DNS interna, l’API usa di default:
- `QDRANT_HOST=qdrant`
- `OLLAMA_URL=http://ollama:11434/api/generate`

Assicurati che i service name nei compose esterni corrispondano (in caso contrario, sovrascrivi questi valori via `.env` o `environment`).

---

## Autenticazione JWT
Tutti gli endpoint protetti richiedono header:
\`\`\`
Authorization: Bearer <JWT_TOKEN>
\`\`\`
- In sviluppo con \`SKIP_AUTH=true\`, qualsiasi token passa la verifica.  
- In produzione, viene validato contro l’endpoint Django specificato.

---

## Documentazione & Swagger
- **Swagger UI**: \`http://<IP>:9000/docs\`  
- **ReDoc**:         \`http://<IP>:9000/redoc\`  
- **OpenAPI JSON**: \`http://<IP>:9000/openapi.json\`  

---

## Endpoints principali

### GET /
Verifica lo stato dell’app:
\`\`\`json
{ "message": "I-NEST_API is running!" }
\`\`\`

### POST /upload
Carica e indicizza un documento (il file resta disponibile in `uploads/` e viene registrato nel database SQLite).  
- Headers: `Authorization: Bearer <token>`  
- Form data:  
  - `file` (obbligatorio)  
  - `username`, `collection` (opzionali)  
  - Formati ammessi: `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`  
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
Elenca la cronologia degli upload memorizzati in SQLite (filtrabile per utente o collection).  
- Headers: `Authorization: Bearer <token>`  
- Query params opzionali:  
  - `username`, `collection`, `status`, `upload_id`  
  - `limit` (default 100, massimo 500)  

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
Interroga l'IA sul contenuto.
- Headers: `Authorization: Bearer <token>`
- Form data:
  - `question` (obbligatorio)
  - `username`, `collection` (opzionali)
  - `conversation_history` (opzionale): stringa JSON con lista di turni `{role, content}` oppure stringa libera concatenata (solo per pre-caricare lo storico al primo messaggio)
  - `conversation_id` (opzionale): se presente, appende la domanda/risposta a quella conversazione; se assente, ne crea una nuova

**Risposta**:
\`\`\`json
{
  "message": "Risposta generata dal modello...",
  "conversation_id": "b5742e9c-..."
}
\`\`\`

### GET /conversations
Restituisce le ultime conversazioni di un utente (default ultime 10), ordinate per `updated_at` discendente.
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale, default utente autenticato), `limit` (default 10, max 50)

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
Restituisce i messaggi di una conversazione (in ordine cronologico).
- Headers: `Authorization: Bearer <token>`
- Query params: `username` (opzionale, default utente autenticato)

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

### GET /collection
Elenca gli elementi di una collection.  
- Headers: \`Authorization: Bearer <token>\`  
- Query params: \`username\`, \`collection\`, \`limit\` (default 100)  

**Risposta**:
\`\`\`json
{ "collection": "user_coll", "count": 5, "items": [ {...}, {...} ] }
\`\`\`

### GET /collections
Elenca tutte le collections presenti in Qdrant.  
- Headers: \`Authorization: Bearer <token>\`  
- Query params: \`username\` (opzionale, default tutte le collection). Se valorizzato, vengono restituite solo le collection appartenenti a quello username: nome esatto \`username\` o prefisso \`username_\`.

**Risposta**:
\`\`\`json
{ "count": 2, "collections": ["user_a", "org_b_docs"] }
\`\`\`

### DELETE /collection
Elimina una collection.  
- Headers: \`Authorization: Bearer <token>\`  
- Query params: \`username\`, \`collection\`  

**Risposta**:
\`\`\`json
{ "message": "Collection 'user_coll' eliminata con successo." }
\`\`\`

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
2. Branch \`feature/x\`, \`bugfix/y\`  
3. Test con \`pytest\`  

---

## Licenza
MIT License

## TEST Rapido
## #############################################
ps -u stefani -f | grep '[u]vicorn'

uvicorn app:app   --host 0.0.0.0   --port 9000
nohup uvicorn app:app --host 0.0.0.0 --port 9000 > uvicorn.log 2>&1 &

Utenze:
pc casaccia: demonstration:mosaico1234
laptop:dimostrazione:mosaico1234
callia: dimostrazione:mosaico1234

## #############################################
