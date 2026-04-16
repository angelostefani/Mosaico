# Flusso applicativo e interazione componenti — Progetto Mosaico

## Obiettivo

Ecosistema modulare per ricerca e chat RAG su documenti aziendali. Comprende:

- **Frontend Django** (mosaico, porta 9001): UI per login/registrazione, upload, chat privata/pubblica, storico caricamenti e configurazione collection.
- **API FastAPI** (ai_api, porta 9000): pipeline ingest → embedding → Qdrant → risposta LLM, persistenza metadati e gestione conversazioni.
- **Qdrant** (porta 6333): database vettoriale per i chunk indicizzati.
- **Ollama** (porta 11434): serving LLM locale (modello di default configurabile, es. `gemma3:1b`).
- **SQLite** (default) o **Postgres** (porta host 5433 → container 5432, opzionale): metadati upload e cronologia conversazioni.

## Architettura logica

```
Browser
  │  sessione + form
  ▼
Django Frontend (:9001)
  │  REST con Authorization: Bearer <jwt>
  │  API_BASE (default http://xxx.xxx.xxx.xxx:9000)
  ▼
FastAPI (ai_api) (:9000)
  ├──► Qdrant (:6333)   — insert/search vettori
  ├──► Ollama (:11434)  — generazione risposta
  └──► SQLite/Postgres  — metadati upload, conversazioni
```

- Il browser non contatta mai direttamente l'API FastAPI: tutte le chiamate passano per il Django server-side (tranne le chiamate JS client-side a `/ollama/models`, `/collections`, ecc., che usano `FAKE_TOKEN`).
- I tenant sono isolati tramite il prefisso `<username>_<collection>` nel nome collection Qdrant.
- JWT lifetime: 60 minuti. In dev è possibile usare `FAKE_TOKEN` come fallback o impostare `SKIP_AUTH=true` nell'API.

## Flusso end-to-end

### 1. Autenticazione

1. Utente accede a `http://localhost:9001/login/`.
2. Django autentica l'utente con sessione locale e richiede JWT a `/api/token/` (DRF SimpleJWT).
3. Il token `access` viene salvato in `request.session['jwt_token']`.
4. In caso di errore JWT, viene salvato un warning in sessione (`jwt_token_error`) e viene usato `FAKE_TOKEN` come fallback.
5. `CSRF_TRUSTED_ORIGINS` e CORS sono configurati per le origini autorizzate.

### 2. Upload documento

1. UI (`/upload/`) invia form con file, `username`, `collection`.
2. Django esegue `POST {API_BASE}/upload` con multipart e header `Authorization: Bearer <token>`; timeout 300 secondi.
3. FastAPI:
   - Valida il file (limite 20 MB), salva su `uploads/`.
   - Estrae testo, applica chunking (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`).
   - Genera embedding con `sentence-transformers`.
   - Inserisce punti in Qdrant nella collection `<username>_<collection>`.
   - Registra record in DB con `status`, `num_chunk`, `filename`, `upload_id`.
4. Risposta: `upload_id`, collection usata, punti inseriti, filename salvato.
5. La UI mostra l'esito e propone un link diretto alla chat con la stessa collection.

### 3. Chat RAG

1. UI (`/chat/` autenticata o `/public-chat/` con `FAKE_TOKEN`) invia `POST {API_BASE}/chat/stream` con:
   - `question`, `username`, `collection` (anche vuota).
   - `model` selezionato (opzionale).
   - `conversation_id` e `conversation_history` (ultimi N turni entro `CHAT_HISTORY_CHAR_BUDGET` caratteri).
2. FastAPI:
   - Calcola embedding della domanda (con espansione multi-vector opzionale).
   - Ricerca Qdrant (candidati `CHAT_CANDIDATES=50`, soglia adattiva: default `QDRANT_SCORE_THRESHOLD=0.42` con fallback progressivo × 0.75 → × 0.5 → 0.0).
   - Reranking ibrido (vector + SequenceMatcher + keyword), opzionale cross-encoder (`ENABLE_CROSS_ENCODER_RERANK`).
   - MMR diversity selection (`MMR_LAMBDA` configurabile, default 0.65).
   - Context stitching sentence-aware (`ENABLE_STITCH`), troncamento al limite del budget senza spezzare frasi.
   - Costruisce prompt e chiama Ollama in streaming SSE; risposta inviata token per token.
   - Salva turni in DB.
3. Risposta SSE: `data: {"chunk":"..."}` progressivi + `data: {"done":true,"conversation_id":"..."}`.
4. La history è mantenuta anche in `localStorage` lato client.

### 4. Storico upload e gestione collection

- `GET {API_BASE}/uploads?username=&collection=&limit=&offset=` → lista upload paginata con stato e metadati; `count` nella risposta per la paginazione client.
- `GET {API_BASE}/collections` → elenco collection Qdrant dell'utente.
- `GET/PUT/DELETE {API_BASE}/collection/config` → scope_prompt per collection.
- Il frontend normalizza i nomi collection aggiungendo/rimuovendo il prefisso `<username>_`.

### 5. Conversazioni

- `GET {API_BASE}/conversations` → lista conversazioni recenti.
- `GET {API_BASE}/conversations/{id}` → dettaglio con history messaggi.
- La UI permette di aprire una conversazione esistente o crearne una nuova azzerando lo stato locale.

### 6. Migrazione a Postgres (opzionale)

Script `migrate_sqlite_to_postgres.py` legge `uploads.sqlite3` e scrive su Postgres (`postgres_project/docker-compose.yml` esposto su 5433). Upsert su upload/conversazioni con deduplica messaggi; supporta `--dry-run` e `--batch-size`.

## Deployment e networking

I progetti sono indipendenti e vengono avviati singolarmente con Docker Compose.

**Ordine avvio consigliato**: Ollama → Qdrant → API (ai_api) → Postgres (se usato) → Frontend (mosaico)

| Progetto | Compose | Porta | Note |
|---|---|---|---|
| Ollama | `ollama_project/docker-compose.yml` | 11434 | Volume `ollama_data` |
| Qdrant | `qdrant_project/docker-compose.yml` | 6333 | Volume `qdrant_storage` |
| API | `ai_api/docker-compose.yml` | 9000 | `QDRANT_HOST=qdrant`, `OLLAMA_URL=...` |
| Frontend | `mosaico/docker-compose.yml` | 9001 | `API_BASE=http://xxx.xxx.xxx.xxx:9000` |
| Postgres | `postgres_project/docker-compose.yml` | 5433→5432 | Volume `postgres_data` |

Il `docker-compose.yml` del frontend monta `./db.sqlite3` e `./logs` come volumi per la persistenza dei dati tra i restart.

> **Linux**: `extra_hosts: host.docker.internal:host-gateway` è incluso nel compose per la risoluzione dell'host.

## Componenti e responsabilità

| Componente | Responsabilità |
|---|---|
| **Django frontend** | UI, sessione, gestione JWT, i18n IT/EN (`ui/i18n.py`, context processor), sincronizzazione collection tramite `localStorage`, paginazione storico, chiamate REST con header `Authorization` |
| **FastAPI (ai_api)** | Endpoint `/upload`, `/chat`, `/chat/stream`, `/uploads`, `/collection(s)`, `/collection/config`, `/conversations*`, `/healthz`, `/ollama/models`; orchestrazione RAG e persistenza |
| **Qdrant** | Storage vettoriale chunk con payload (`user`, `collection`, `titolo`, `pagina`, `upload_id`) |
| **Ollama** | Generazione testo da contesto RAG; modello configurabile; streaming token-per-token |
| **SQLite/Postgres** | Tracciamento upload, conversazioni, messaggi |

## Sicurezza e operatività

- **CORS**: `CORS_ALLOW_ALL_ORIGINS = True` nel frontend (accettabile in dev). In produzione restringere anche lato API (`allow_origins`).
- **JWT**: impostare `SKIP_AUTH=false` nell'API e `DJANGO_VERIFY_URL` valido per la verifica remota.
- **Secret Key**: sostituire sempre `DJANGO_SECRET_KEY` in produzione.
- **Logging**: rotazione configurabile (`LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`); log SQL opzionale via `LOG_SQL=true`.
- **Health-check API**: `GET /healthz` verifica connettività verso Qdrant, Ollama e storage.
- **Backup**: volumi `ollama_data`, `qdrant_storage`, `postgres_data` salvano rispettivamente modelli, vettori e DB.

## Internazionalizzazione (i18n)

La lingua dell'interfaccia è selezionabile dall'utente tramite il pulsante in navbar. Il meccanismo è interamente custom (senza gettext) per evitare dipendenze GNU su Windows:

| File | Ruolo |
|---|---|
| `ui/i18n.py` | Dizionari `TRANSLATIONS['it']` e `TRANSLATIONS['en']` con ~150 chiavi |
| `ui/context_processors.py` | Legge `request.session['_language']`, inietta `lang` e `t` nel contesto di ogni template |
| `mosaico/settings.py` | Registra `ui.context_processors.i18n` in `TEMPLATES.OPTIONS.context_processors` |
| `ui/views.py` → `set_language` | `POST /set-language/` salva la lingua in sessione e reindirizza |
| Template HTML | Usano `{{ t.chiave }}` per le stringhe server-side |
| Template JS | Usano `const UI = { key: "{{ t.chiave }}" }` per le stringhe dinamiche |

## Riferimenti sorgente

- Frontend e UX: `mosaico/docs/mosaico_documentazione.md`
- API e pipeline RAG: `ai_api/docs/documentazione_progetto.md`
- Infrastruttura container: `ollama_project/docker-compose.yml`, `qdrant_project/docker-compose.yml`, `postgres_project/docker-compose.yml`
- Variabili d'ambiente: `mosaico/.env.example`
