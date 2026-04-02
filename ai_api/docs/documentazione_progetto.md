# Documentazione progetto – Document QA & Chat API
**Versione 1.3 – 16/03/2026**

## Obiettivo
Servizio FastAPI che abilita caricamento, indicizzazione vettoriale e interrogazione conversazionale di documenti tramite pipeline RAG basata su Qdrant e Ollama, con gestione multi-tenant, tracciamento degli upload e supporto dual-database (SQLite / PostgreSQL).

---

## Funzionalità principali

- **Upload e normalizzazione**: accetta file `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, pulisce il testo, effettua chunking configurabile e conserva riferimenti a pagina/offset (o riga/foglio per Excel).
- **Indicizzazione vettoriale**: calcola embedding con `sentence-transformers` e inserisce i punti in Qdrant con payload esteso (`upload_id`, `user`, `collection`, titolo documento, numero chunk/pagina). Il fallback deterministico è disabilitato di default e va abilitato esplicitamente solo per test/debug con `ALLOW_EMBEDDING_FALLBACK=true`.
- **Gestione upload**: salvataggio fisico in `uploads/` (o `UPLOAD_DIR`), metadati nel database con stati `processing/completed/failed` e messaggi di errore.
- **Database duale**: `DB_ENGINE=sqlite` (default, sviluppo) oppure `DB_ENGINE=postgres` (produzione). Lo schema viene creato automaticamente all'avvio in entrambi i casi.
- **Ricerca e chat RAG**: embed della domanda + eventuali espansioni multi-vettore, ricerca in Qdrant, filtro su punteggio minimo, rerank opzionale, dedup e stitching dei chunk per costruire il contesto passato al modello Ollama.
- **Override modello per-request**: `POST /chat` accetta un campo `model` opzionale per usare un modello Ollama diverso da quello di default.
- **Collections multi-tenant**: nomi costruiti come `username_collection`, oppure solo `username` o `collection`, default `documents`. Endpoints per elencare, ispezionare e cancellare collections.
- **Configurazione collection** (`scope_prompt`): prompt di specializzazione per-collection, persistito in DB e applicato automaticamente a `POST /chat`.
- **Cronologia chat**: salvataggio conversazioni con elenco e ripresa tramite `conversation_id`. Numero massimo di conversazioni elencate configurabile via `MAX_CONVERSATION_LIST`.
- **Elenco modelli Ollama**: `GET /ollama/models` interroga l'endpoint `/api/tags` di Ollama e restituisce i modelli disponibili.
- **Autenticazione e logging**: JWT validato contro Django (`DJANGO_VERIFY_URL`) con bypass `SKIP_AUTH` per sviluppo/test; logging con request-id, rotazione file (`LOG_FILE`) e CORS configurabile via `CORS_ORIGINS`.
- **Monitoraggio**: health-check `/healthz` (Qdrant, Ollama, database, storage + config runtime), frontend statico su `/static`.
- **Migrazione dati**: script `migrate_sqlite_to_postgres.py` per spostare upload e conversazioni da SQLite a Postgres con upsert e deduplica messaggi.

---

## Architettura e componenti

```
app.py           — FastAPI application, tutti gli endpoint, logica RAG
db.py            — Modelli SQLAlchemy (Upload, Conversation, ConversationMessage, CollectionConfig)
                   + factory create_engine_and_session() con supporto SQLite/Postgres
migrate_sqlite_to_postgres.py — script di migrazione one-shot
tests/           — suite pytest
debug/           — script di debug stand-alone (chat, upload, API)
frontend/        — asset statici serviti su /static
uploads/         — file caricati + uploads.sqlite3 (se DB_ENGINE=sqlite)
logs/            — log applicativi con rotazione
```

---

## Modello dati (db.py)

| Tabella | Chiave | Campi principali |
|---|---|---|
| `uploads` | `upload_id` (UUID) | `username`, `collection`, `original_filename`, `stored_filename`, `size_bytes`, `num_chunks`, `num_points`, `status`, `error_message` |
| `conversations` | `conversation_id` (UUID) | `user_id`, `title`, `collection`, `created_at`, `updated_at` |
| `conversation_messages` | `id` | `conversation_id` (FK), `role`, `content`, `created_at` |
| `collection_configs` | `collection_name` | `scope_prompt`, `created_at`, `updated_at` |

---

## Flussi operativi

### Caricamento documento
1. `POST /upload` riceve file + opzionali `username` e `collection`; salva il file con nome UUID (limite `MAX_UPLOAD_SIZE_BYTES`, default 20 MB).
2. Estrazione testo (anche paginata per PDF, per foglio per Excel), normalizzazione, chunking e embedding.
3. Creazione/uso della collection Qdrant, inserimento punti con metadata e aggiornamento record nel database.
4. Risposta con riepilogo: chunks inseriti, collection usata, `upload_id`, nome file salvato e titolo documento derivato.

### Conversazione RAG
1. `POST /chat` valida la collection target e calcola embedding per domanda + termini di espansione (se `ENABLE_MULTI_VECTOR_SEARCH`).
2. Query a Qdrant (primaria + opzionali espansioni), filtro per `QDRANT_SCORE_THRESHOLD`, rerank opzionale (`ENABLE_RERANK`), MMR opzionale (`ENABLE_MMR`), dedup e stitching rank-preserving entro `CHAT_CONTEXT_CHAR_BUDGET`.
3. Se la collection ha un `scope_prompt` configurato, viene preposto al system prompt.
4. Composizione prompt con cronologia (JSON di turni o testo libero) e invio a Ollama (`OLLAMA_URL`, modello `OLLAMA_MODEL` o override `model` per-request).
5. Salvataggio messaggi nel database con `conversation_id` restituito nella risposta (riutilizzabile per continuare la conversazione).

---

## Endpoints esposti

| Metodo | Path | Descrizione | Auth |
|---|---|---|---|
| `GET` | `/` | Ping servizio | No |
| `POST` | `/upload` | Upload e indicizzazione documento | Sì |
| `GET` | `/uploads` | Elenco upload, filtrabile per `username`, `collection`, `status`, `upload_id`; `limit` default 100 (max 500) | Sì |
| `POST` | `/chat` | Chat RAG; opz. `username`, `collection`, `conversation_history`, `conversation_id`, `model` | Sì |
| `GET` | `/conversations` | Ultime conversazioni per utente (default 10, max `MAX_CONVERSATION_LIST`) | Sì |
| `GET` | `/conversations/{id}` | Messaggi di una conversazione in ordine cronologico | Sì |
| `GET` | `/collection` | Elementi di una collection (payload + id), `limit` default 100 | Sì |
| `GET` | `/collections` | Elenco collections da Qdrant, filtrabile per `username` | Sì |
| `DELETE` | `/collection` | Cancella la collection (e la config associata) | Sì |
| `GET` | `/collection/config` | Legge la configurazione scope_prompt di una collection | Sì |
| `PUT` | `/collection/config` | Crea o aggiorna scope_prompt di una collection | Sì |
| `DELETE` | `/collection/config` | Elimina la configurazione di una collection | Sì |
| `GET` | `/ollama/models` | Elenco modelli disponibili su Ollama | Sì |
| `GET` | `/healthz` | Stato Qdrant / Ollama / database / storage + config runtime | No |
| statico | `/static/index.html` | Frontend servito dalla cartella `frontend/` | No |

---

## Configurazione essenziale (.env)

| Gruppo | Variabili chiave |
|---|---|
| Storage | `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_BYTES` |
| Database | `DB_ENGINE` (`sqlite`\|`postgres`), `UPLOAD_DB_PATH` (sqlite), `PG_HOST/PORT/DB/USER/PASSWORD/SSLMODE` (postgres) |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_SCORE_THRESHOLD` |
| Embedding/RAG | `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `ENABLE_RAG_DEBUG`, `ENABLE_RERANK`, `ENABLE_MMR`, `ENABLE_STITCH`, `ENABLE_MULTI_VECTOR_SEARCH`, `CHAT_RESULT_LIMIT`, `CHAT_CANDIDATES`, `CHAT_EXPANSION_LIMIT`, `CHAT_EXPANSION_CANDIDATES`, `CHAT_CONTEXT_CHAR_BUDGET`, `CHAT_HISTORY_PROMPT_LIMIT` |
| Ollama | `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` |
| Auth | `DJANGO_VERIFY_URL`, `SKIP_AUTH` |
| Conversazioni | `CONVERSATION_PREVIEW_CHARS`, `MAX_CONVERSATION_LIST` |
| CORS | `CORS_ORIGINS` (lista separata da virgola, `*` per sviluppo) |
| Logging | `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT` |

### Valori di default (marzo 2026)
- `DB_ENGINE=sqlite`
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`
- `QDRANT_SCORE_THRESHOLD=0.3`
- `CHAT_RESULT_LIMIT=30`, `CHAT_CANDIDATES=30`, `CHAT_CONTEXT_CHAR_BUDGET=7000`
- `OLLAMA_MODEL=gpt-oss:20b`, `OLLAMA_TIMEOUT=60s`
- `MAX_UPLOAD_SIZE_BYTES=20_000_000` (20 MB)
- `CONVERSATION_PREVIEW_CHARS=120`, `MAX_CONVERSATION_LIST=50`
- `CORS_ORIGINS=http://localhost:9001`

---

## Persistenza e asset

- File caricati e (se `DB_ENGINE=sqlite`) database SQLite nella directory `UPLOAD_DIR`.
- Vettori e payload in Qdrant, separati per tenant/collection.
- Conversazioni e configurazioni collection nel database (SQLite o Postgres).
- Log applicativi con rotazione in `logs/` (percorso configurabile).
- Frontend statico in `frontend/` pubblicato su `/static`.

---

## Docker Compose

Il `docker-compose.yml` avvia solo l'API e si collega a due reti Docker esterne:
- `qdrant_project_default` — dove è attivo il container Qdrant
- `postgres_project_default` — dove è attivo il container PostgreSQL

Ollama viene raggiunto tramite `host.docker.internal` (processo sul host o container su un altro stack separato).

Variabili iniettate dal compose:
```
QDRANT_HOST=qdrant          (service name sulla rete qdrant_external)
OLLAMA_URL=http://host.docker.internal:11434/api/generate
DB_ENGINE=postgres
PG_HOST=postgres            (service name sulla rete postgres_external)
```

Le credenziali Postgres (`PG_DB`, `PG_USER`, `PG_PASSWORD`) vengono lette dal file `.env`.

---

## Note operative e test

- Middleware di logging aggiunge `request-id` e tempi di risposta a ogni richiesta.
- CORS configurabile via `CORS_ORIGINS`; in sviluppo si può usare `*`.
- Se il modello embedding non è disponibile e `ALLOW_EMBEDDING_FALLBACK=false`, il servizio resta avviato ma gli endpoint RAG (`/upload`, `/chat`, `/chat/stream`) rispondono `503` e `/healthz` segnala stato degradato.
- Suite di test: `pytest -q` (copre chunking, estrazione, endpoints principali e casi mirati di stitching, rerank/MMR e fallback embedding).
- Script di debug in `debug/` per testare manualmente upload, chat e API da riga di comando.
