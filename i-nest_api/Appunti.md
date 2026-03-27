Descrizione di cosa fa l’applicazione.

In Sintesi

- Document QA RAG: indicizza documenti e risponde a domande basandosi sui contenuti.
- Pipeline: upload file → estrazione testo → chunking → embedding → salvataggio in Qdrant → ricerca semantica →
- Multi-tenant: separa i dati per utente/collection costruendo il nome della collection (es. username_collection).
- Auth: JWT con verifica delegata a Django; in sviluppo può saltare la verifica (SKIP_AUTH=true).
- Frontend statico: serve una semplice pagina su /static.

Tecnologie

- API: FastAPI (+ CORS).
- Vettori: sentence-transformers (all-MiniLM-L6-v2).
- Vector DB: Qdrant (cosine distance).
- LLM: Ollama (modello gemma3:1b via OLLAMA_URL).
- Parser: pdfplumber per PDF, python-docx per DOC/DOCX, lettura diretta per TXT.

Endpoint Principali

- GET /: health check.
- POST /upload: carica e indicizza un documento (JWT richiesto).
- POST /chat: domanda sul contenuto indicizzato, con recupero top-k da Qdrant e risposta generata (JWT).
- GET /collection: lista elementi di una collection (JWT).
- DELETE /collection: elimina una collection (JWT).
- GET /static/*: serve il frontend statico.

Configurazione (.env)

- UPLOAD_DIR, QDRANT_HOST/PORT, OLLAMA_URL, EMBEDDING_MODEL, CHUNK_SIZE.
- DJANGO_VERIFY_URL, SKIP_AUTH per la gestione dell’autenticazione.