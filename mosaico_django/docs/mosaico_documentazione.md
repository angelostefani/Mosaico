# Mosaico — traccia slide

Sintesi per presentare il funzionamento dell'applicativo Django + FastAPI che gestisce upload di documenti, interrogazioni via chat e consultazione delle collection.

## Schermate

### Login (`/login`)
- Form Django standard (username, password) con validazione lato server.
- Dopo autenticazione crea sessione Django e chiede un JWT a `/api/token/`; se ottenuto viene salvato in sessione per le chiamate API.
- Messaggi di errore inline e link rapido alla registrazione.

### Registrazione (`/register`)
- Form di creazione utente (`UserCreationForm`), controlla password e duplicati.
- Al salvataggio effettua login automatico; tenta l'ottenimento del JWT e, se fallisce, mostra un avviso persistente nella sessione.
- Link di ritorno al login.

### Home (`/`)
- Hero con descrizione di Mosaico e call-to-action verso Upload e Chat.
- Tre card riassuntive: Storico Uploads, Upload guidato, Chat potenziata (link diretti alle rispettive viste).
- Navbar comune con accesso a Upload, Chat, Caricamenti e gestione utente/log-out.

### Upload documento (`/upload`)
- Form client-side (JS) con input file, collection, utente precompilato (sola lettura) e link verso la chat con la stessa collection.
- Validazione sul nome collection (niente underscore), salvataggio/sincronizzazione su `localStorage` per condivisione tra pagine.
- Richiesta `POST ${API_BASE}/upload` con `Authorization: Bearer <jwt|FAKE_TOKEN>`; feedback immediato di successo/errore e ID upload quando disponibile.
- Avviso se il JWT non e' stato ottenuto in fase di login/registrazione.

### Chat autenticata (`/chat`)
- Finestra conversazione con badge dinamico della collection, input domanda e pulsante invio (spinner e disabilitazione durante la chiamata).
- Form contestuale: collection (sincronizzata con Upload via `localStorage`), utente precompilato, link rapido alla pagina di upload con stessa collection.
- Richiesta `POST ${API_BASE}/chat` con token in header; mostra risposta o errore nell'area conversazione.

### Storico caricamenti (`/uploads`)
- Filtri: utente (precompilato), collection (suffix senza prefisso utente), limite risultati.
- Costruzione del nome collection lato client aggiungendo il prefisso `<username>_` quando mancante; rimozione del prefisso in tabella per leggibilita'.
- Chiamata `GET ${API_BASE}/uploads?username=...&collection=...&limit=...` con token; tabella dinamica con ID, utente, collection, file, stato (badge), chunks e timestamp.
- Messaggi di errore e riepilogo conteggi; caricamento automatico se arrivano parametri in query string.

### Chat pubblica (`/public-chat/?username=...&collection=...`)
- Accesso senza login; richiede `username` e `collection` in query string e li blocca lato client.
- UI full-screen indipendente dal layout principale, con hero statico, card meta info, conversazione, form domanda, hint tastiera e badge di stato chiamata API.
- Richiesta `POST ${API_BASE}/chat` con `Authorization: Bearer <FAKE_TOKEN>`; persiste cronologia, errori e status code in sessione utente per rilascio successivi.
- Gestione errori a banner e fallback non-JS con messaggio dedicato.

## Funzionalita'

- **Autenticazione**: sessione Django per la UI; JWT ottenuto da `/api/token/` e salvato in sessione (`request.session.jwt_token`). In assenza, le chiamate usano `FAKE_TOKEN` di configurazione.
- **Gestione token**: header `Authorization: Bearer <token>` per `/upload`, `/chat`, `/uploads`; warning specifico se il JWT manca.
- **Upload documenti**: invio multipart a `/upload` con file, collection e username; feedback in pagina, link di contesto verso la chat con stessa collection.
- **Chat IA**: invio domanda a `/chat` con collection e username; UI mostra spinner, badge collection, cronologia client-side; versione pubblica usa la stessa API senza autenticazione.
- **Storico upload**: interrogazione `GET /uploads` con filtro per utente/collection; normalizzazione nomi, badge di stato, conteggio totale.
- **Sincronizzazione collection**: uso di `localStorage` e di parametri query per mantenere la stessa collection tra Upload, Chat e Storico; link incrociati aggiornano automaticamente le query string.
- **Gestione errori e UX**: messaggi alert per errori API/rete, badge di stato HTTP (chat pubblica), indicatori di caricamento per upload/chat/lista, placeholder e testi guida in ogni form.
- **Configurazione**: `API_BASE` (default `http://localhost:9000`) nel settings Django; `FAKE_TOKEN` per ambienti dev; porte standard 9001 (frontend) e 9000 (API).

Usa questa traccia come base slide: una sezione per le schermate (con screenshot) e una per le funzioni chiave e i flussi API (autenticazione, upload, chat, storico, chat pubblica).
