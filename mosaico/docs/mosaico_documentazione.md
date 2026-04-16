# Mosaico — Documentazione schermate

Sintesi per presentare il funzionamento dell'app Django + FastAPI che gestisce upload di documenti, chat assistita e consultazione delle collection (con supporto per conversazioni salvate, selezione modello e interfaccia bilingue IT/EN).

## Schermate

### Login (`/login/`)
- Form Django con validazione server-side; messaggi di errore inline e link a registrazione.
- Dopo autenticazione crea sessione Django e richiede JWT a `/api/token/`; se ottenuto lo salva in sessione per le chiamate API.
- Redirect automatico alla home al login riuscito.

### Registrazione (`/register/`)
- Usa `UserCreationForm`, controlla password/duplicati.
- Effettua login automatico; tenta il recupero del JWT e, in caso di errore, mostra un avviso persistente in sessione (`jwt_token_error`).

### Home (`/`)
- Hero con descrizione e call-to-action verso Upload e Chat.
- Card riepilogo: Storico Uploads, Upload guidato, Chat potenziata; navbar condivisa con pulsante log-out.
- Richiede autenticazione.

### Upload documento (`/upload/`)
- Layout a due pannelli: pannello sinistro per la selezione/creazione collection, pannello destro con dropzone per i file.
- Modalità collection: toggle "Esistente" / "Nuova"; dropdown caricato da `/collections`; nuovo nome validato (no underscore).
- Supporta selezione multipla di file con progress bar per-file e badge di stato (In attesa / In caricamento / Completato / Errore).
- Footer con link "Vai alla chat con questa collection" (aggiornato dinamicamente) e pulsante "Carica".
- Sincronizza la collection selezionata su `localStorage` con listener `storage` per altre schede.
- Richiesta `POST /upload/` inoltrata via AJAX; il Django view la inoltra a `{API_BASE}/upload` con header `Authorization: Bearer <jwt|FAKE_TOKEN>`; timeout 300 secondi.
- Warning se il JWT non è stato ottenuto in fase di login.

### Chat autenticata (`/chat/`)
- Area conversazione con badge collection, input domanda, spinner e disabilitazione durante la chiamata.
- Form di contesto: collection (caricata da `/collections` e sincronizzata via `localStorage`), utente readonly, selezione modello da `/ollama/models` (persistita su `localStorage`, opzionale).
- Conversazioni recenti: lista da `/conversations` con anteprima, collection e data; pulsante "Nuova" per azzerare lo stato locale.
- Apertura conversazione: click su una card → `GET /conversations/{id}`; ripristina history e collection.
- Invio domanda: `POST ${API_BASE}/chat` con `collection` (anche vuota), `model` selezionato, `conversation_id` corrente e `conversation_history` (ultimi 10 turni) serializzati; salvataggio storico in `localStorage` e highlight conversazione attiva.

### Storico caricamenti (`/uploads/`)
- Filtri: collection (suffix senza prefisso utente) e righe per pagina (10/25/50/100).
- Paginazione server-side: `GET {API_BASE}/uploads?username=...&collection=...&limit=...&offset=...`; componente Bootstrap con ellipsis, info "Pagina X di Y" e riepilogo "Risultati X–Y di Z".
- Normalizza il nome collection aggiungendo il prefisso `<username>_` per la chiamata API; rimuove il prefisso in tabella per leggibilità.
- Tabella con ordinamento client-side (click su intestazione) per le colonne principali.
- Autoload se arrivano parametri in query string (`collection`, `username`).

### Configurazione collection (`/collection-config/`)
- Interfaccia per caricare, salvare ed eliminare lo `scope_prompt` di una collection.
- Dropdown collection caricato da `GET {API_BASE}/collections`; sincronizzato con `localStorage`.
- `GET /collection/config` carica il prompt corrente; `PUT /collection/config` salva; `DELETE /collection/config` elimina con modal di conferma.
- Messaggi di stato inline (successo/errore/warning) senza reload di pagina.

### Chat pubblica (`/public-chat/?username=...&collection=...`)
- Accesso senza login: richiede `username` e `collection` in query string (non modificabili lato client).
- Layout full-screen indipendente, con hero statico, card meta, area conversazione, form domanda, hint tastiera e badge di stato chiamata API.
- `POST ${API_BASE}/chat` con `Authorization: Bearer <FAKE_TOKEN>`; persiste cronologia, errori e status code in sessione Django per ripristino.
- Banner di errore e fallback non-JS (avviso che JavaScript è richiesto).
- Restituisce 400 se `username` o `collection` mancano nella query string.

### Cambio password (`/password/change/`)
- Form Django con validazioni e help testuale; pulsanti Annulla/Salva, errori inline.
- Usa `update_session_auth_hash` per mantenere la sessione valida dopo il cambio.
- Richiede autenticazione.

## Funzionalità trasversali

- **Autenticazione**: sessione Django per la UI; JWT (lifetime 60 min) da `/api/token/` salvato in `request.session['jwt_token']`; fallback `FAKE_TOKEN` in dev.
- **Gestione token**: header `Authorization: Bearer <token>` per tutte le chiamate verso l'API (`/upload`, `/chat`, `/uploads`, `/collections`); warning dedicato se manca JWT.
- **Upload documenti**: dropzone multi-file, progress per-file, timeout 300 s; cross-link verso la chat con la stessa collection.
- **Chat IA**: streaming SSE `/chat/stream` token per token; badge collection, selezione modello, cronologia client-side; versione pubblica senza login.
- **Conversazioni e modelli**: elenco da `/conversations`, apertura dettaglio, nuovo thread; invio include `conversation_id` e `conversation_history` (ultimi 10 turni). Dropdown modelli da `/ollama/models` con persistenza locale.
- **Storico upload**: filtro per collection, paginazione server-side con `offset`, normalizzazione prefisso utente, badge di stato, ordinamento client-side.
- **Sincronizzazione collection**: `localStorage` condiviso tra Upload, Chat e Storico; query string preservate nei link incrociati.
- **i18n IT/EN**: pulsante lingua in navbar, preferenza salvata in `request.session['_language']`; context processor `ui.context_processors.i18n` inietta `lang` e `t` (dizionario da `ui/i18n.py`) in ogni template; blocco JS `UI` renderizzato server-side per le stringhe dinamiche.
- **Gestione errori e UX**: alert per errori API/rete, badge HTTP nella chat pubblica, indicatori di caricamento per ogni form, placeholder e testi guida.
- **Configurazione**: `API_BASE` (default `http://xxx.xxx.xxx.xxx:9000`), `FAKE_TOKEN` per dev, porte 9001 (frontend) e 9000 (API).

## URL di riferimento

| Pagina | URL |
|---|---|
| Home | `http://localhost:9001/` |
| Login | `http://localhost:9001/login/` |
| Registrazione | `http://localhost:9001/register/` |
| Upload | `http://localhost:9001/upload/` |
| Storico upload | `http://localhost:9001/uploads/` |
| Chat | `http://localhost:9001/chat/` |
| Configurazione collection | `http://localhost:9001/collection-config/` |
| Chat pubblica | `http://localhost:9001/public-chat/?username=<utente>&collection=<collection>` |
| Cambio password | `http://localhost:9001/password/change/` |

Usa questa traccia per le slide: sezione schermate (con screenshot) e sezione flussi chiave (autenticazione, upload, chat con conversazioni/modello, storico, collection-config, chat pubblica).
