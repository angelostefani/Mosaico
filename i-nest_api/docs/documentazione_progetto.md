# Documentazione progetto – Document QA & Chat API

## Obiettivo
Servizio FastAPI che abilita caricamento, indicizzazione vettoriale e interrogazione conversazionale di documenti tramite pipeline RAG basata su Qdrant e Ollama, con gestione multi-tenant e tracciamento degli upload.

## Funzionalita principali
- **Upload e normalizzazione**: accetta file `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, pulisce il testo, effettua chunking configurabile e conserva riferimenti a pagina/offset (o riga/foglio per Excel).
- **Indicizzazione vettoriale**: calcola embedding con `sentence-transformers` (fallback deterministico se il modello non e disponibile) e inserisce i punti in Qdrant con payload esteso (upload_id, user, collection, titolo documento, numero chunk/pagina).
- **Gestione upload**: salvataggio fisico in `uploads/` (o `UPLOAD_DIR`), metadati in SQLite (`uploads.sqlite3` o `UPLOAD_DB_PATH`) con stati `processing/completed/failed` e messaggi di errore.
- **Ricerca e chat RAG**: embed della domanda + eventuali espansioni multi-vettore, ricerca in Qdrant, filtro su punteggio minimo, rerank opzionale, dedup e stitching dei chunk per costruire il contesto passato al modello Ollama.
- **Collections multi-tenant**: nomi costruiti come `username_collection`, oppure solo `username` o `collection`, default `documents`. Endpoints per elencare, ispezionare e cancellare collections.
- **Cronologia chat**: salvataggio conversazioni con elenco ultime 10 e ripresa tramite `conversation_id`.
- **Autenticazione e logging**: JWT validato contro Django (`DJANGO_VERIFY_URL`) con bypass `SKIP_AUTH` per sviluppo/test; logging con request-id, rotazione file (`LOG_FILE`) e CORS aperto.
- **Monitoraggio**: health-check /healthz (Qdrant, Ollama, storage), elenco modelli Ollama e statico frontend servito su `/static`.

## Flussi operativi
### Caricamento documento
1. POST `/upload` riceve file + opzionali `username` e `collection`; salva il file con nome UUID (limite `MAX_UPLOAD_SIZE_BYTES`, default 20 MB).
2. Estrazione testo (anche paginata per PDF), normalizzazione, chunking e embedding.
3. Creazione/uso della collection Qdrant, inserimento punti con metadata e aggiornamento record SQLite.
4. Risposta con riepilogo: chunks inseriti, collection usata, upload_id, nome file salvato e titolo documento derivato.

### Conversazione RAG
1. POST `/chat` valida la collection target e calcola embedding per domanda + termini di espansione (se `ENABLE_MULTI_VECTOR_SEARCH`).
2. Query a Qdrant (primaria + opzionali espansioni), filtro per `QDRANT_SCORE_THRESHOLD`, rerank opzionale (`ENABLE_RERANK`, `ENABLE_MMR`), dedup e stitching entro `CHAT_CONTEXT_CHAR_BUDGET`.
3. Composizione prompt con cronologia (JSON di turni o testo libero) e invio a Ollama (`OLLAMA_URL`, modello `OLLAMA_MODEL` o override per-request). Salvataggio messaggi in SQLite con `conversation_id` restituito nella risposta (riutilizzabile per continuare una conversazione). Risposta testuale nel campo `message`.

## Endpoints esposti
- `GET /` – ping servizio.
- `POST /upload` - upload e indicizzazione documento. Form fields: `file`, opz. `username`, `collection`. Richiede Authorization salvo `SKIP_AUTH`. Formati ammessi: `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`.
- `GET /uploads` – elenco upload filtrabile per `username`, `collection`, `status`, `upload_id`; limite default 100 (max 500).
- `GET /collection` - elementi di una collection (payload + id) con limite configurabile.
- `GET /collections` - elenco collections da Qdrant, filtrabile per `username`.
- `DELETE /collection` - cancella la collection calcolata (tenant-aware).
- `POST /chat` - domanda al chatbot con opz. `username`, `collection`, `conversation_history` (JSON lista turni o testo), `conversation_id` per continuare chat esistente, `model` per override. Risponde con `message` e `conversation_id`.
- `GET /conversations` - ultime conversazioni (default 10) per utente, ordinate per `updated_at`.
- `GET /conversations/{conversation_id}` - messaggi di una conversazione in ordine cronologico.
- `GET /ollama/models` - elenco modelli disponibili da Ollama (/api/tags).
- `GET /healthz` – stato Qdrant/Ollama/storage e configurazione runtime.
- Statico: `/static/index.html` servito dalla cartella `frontend/`.

## Configurazione essenziale (.env)
- Storage: `UPLOAD_DIR`, `UPLOAD_DB_PATH`, `MAX_UPLOAD_SIZE_BYTES`.
- Qdrant: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_SCORE_THRESHOLD`.
- Embedding: `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `ENABLE_RAG_DEBUG`, `ENABLE_RERANK`, `ENABLE_MMR`, `ENABLE_STITCH`, `ENABLE_MULTI_VECTOR_SEARCH`, `CHAT_RESULT_LIMIT`, `CHAT_CANDIDATES`, `CHAT_EXPANSION_LIMIT`, `CHAT_EXPANSION_CANDIDATES`, `CHAT_CONTEXT_CHAR_BUDGET`.
- Ollama: `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`.
- Auth: `DJANGO_VERIFY_URL`, `SKIP_AUTH`.
- Logging: `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`.

## Persistenza e asset
- File caricati e database SQLite nella directory indicata da `UPLOAD_DIR`.
- Vettori e payload in Qdrant, separati per tenant/collection.
- Log applicativi con rotazione in `logs/` (percorso configurabile).
- Frontend statico in `frontend/` pubblicato su `/static`.

## Note operative e test
- Middleware di logging aggiunge request-id e tempi di risposta; CORS aperto per sviluppo.
- Suite di test: `pytest -q` (copre chunking, estrazione, endpoints principali e integrazione con fallback embedding).
