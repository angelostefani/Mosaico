# I-NEST API & Front-End Mosaico

Progetto: servizi REST in FastAPI con autenticazione JWT verificata via Django e front-end Django + Bootstrap per interagire con le API.

## Architettura

- Frontend (Django) — questo repository — espone `http://localhost:9001`.
- API REST — container separato — espone `http://localhost:9000` (avviato dal suo progetto/compose).
- Ollama — container separato — porta predefinita `11434` (progetto dedicato).
- Qdrant — container separato — porta predefinita `6333` (progetto dedicato).

Networking e configurazione:
- Browser → Frontend (`:9001`).
- Frontend (server-side) → API tramite `API_BASE` (default `http://host.docker.internal:9000` nel compose).
- API → Ollama e Qdrant secondo i rispettivi `docker-compose.yml` dei loro progetti.
- CORS/CSRF: il frontend include `CSRF_TRUSTED_ORIGINS` per `localhost:9001`; l’API deve consentire CORS dal frontend.

## Caratteristiche

* **FastAPI** per esposizione di endpoint:

  * `/upload`       : upload e indicizzazione documenti (PDF, DOCX)
  * `/chat`         : interazione chat basata su Document QA
  * `/collection`   : elenco item di una collection in Qdrant
  * `/delete`       : cancellazione di una collection opzionale
* **Autenticazione JWT** con verifica remota a Django
* **Django** front-end:

  * login/registrazione/logout utenti
  * salvataggio JWT in sessione Django
  * interfaccia `upload.html`, `chat.html`, `list.html`
  * template e layout Bootstrap responsive
* **CORS** configurato in FastAPI per consentire chiamate cross-origin da `localhost:9001`

## Prerequisiti

* Python 3.11+
* Docker & Docker Compose (opzionale)
* Virtualenv consigliato

## Setup FastAPI (Backend)

1. Clona il repository.
2. Crea e attiva un virtualenv:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .\.venv\Scripts\activate  # Windows
   ```
3. Installa dipendenze:

   ```bash
   pip install -r requirements.txt
   ```
4. Configura variabili d'ambiente (.env):

   ```dotenv
   DJANGO_VERIFY_URL=http://localhost:9001/api/token/verify/
   SKIP_AUTH=false         # oppure true in sviluppo
   # CORS_ORIGINS=...      # opzionale se NON usa allow_origins=['*']
   ```
5. Avvia con Uvicorn/Hypercorn:

   ```bash
  uvicorn mosaico.asgi:application --host 0.0.0.0 --port 9001 --reload
   
  nohup uvicorn mosaico.asgi:application --host 0.0.0.0 --port 9001 --reload > mosaico.log 2>&1 &

   ```

### Documentazione Swagger

* Swagger UI: [http://localhost:9000/docs](http://localhost:9000/docs)
* OpenAPI JSON:  [http://localhost:9000/openapi.json](http://localhost:9000/openapi.json)

## Setup Django (Front-End & Auth)

1. Dalla cartella del progetto, attiva il virtualenv se serve.
2. Installa le dipendenze comuni:

   ```bash
   pip install -r requirements.txt
   ```
3. Copia il file di esempio e personalizza le variabili:

   ```bash
   cp .env.example .env  # Windows: copy .env.example .env
   ```

   Variabili disponibili:
   - `DJANGO_SECRET_KEY`: chiave Django (obbligatoria in produzione).
   - `DJANGO_DEBUG`: `true/false` per l'ambiente.
   - `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`: host/URL autorizzati.
   - `API_BASE`: endpoint dell'API FastAPI.
   - `FAKE_TOKEN`: token di servizio per chiamate verso l'API durante lo sviluppo.
4. Migrazioni e creazione superuser:

   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```
5. Avvia il server:

   ```bash
   python manage.py runserver 0.0.0.0:9001
   ```
6. Endpoint JWT Django:

   * POST `/api/token/`          : ottieni `access` & `refresh`
   * POST `/api/token/refresh/`  : rinnova token
   * POST `/api/token/verify/`   : verifica token

## Utilizzo

1. **Registrazione/Login**: accedi su `http://localhost:9001/register` o `login`.
2. **Upload**: vai a `http://localhost:9001/upload`, scegli un file e clicca Upload.
3. **Chat**: `http://localhost:9001/chat`, invia domande e leggi risposte.
4. **Collection**: `http://localhost:9001/collection`, inserisci parametri e premi Elenca.
5. **Chat pubblica**: `http://localhost:9001/public-chat/?username=<utente>&collection=<collection>` permette l’accesso senza autenticazione e interroga l’API con i parametri forniti.

Tutte le chiamate dal front-end includono automaticamente il JWT negli header `Authorization: Bearer <token>`.

## Docker (Frontend separato)

Il frontend Django gira in un container separato; API, Ollama e Qdrant restano nei loro progetti/compose.

- Prerequisiti: avvia API (porta `9000`), Ollama e Qdrant con i loro `docker-compose.yml` nei rispettivi progetti.
- Dal progetto `mosaico`:
  - Prod-like: `docker compose up -d --build frontend`
  - Dev (hot-reload): `docker compose --profile dev up -d --build frontend-dev`
  - Apri: `http://localhost:9001`

Ambiente chiave (già nel compose):
- `API_BASE=http://host.docker.internal:9000` per chiamate server-side verso l'API sull’host.
- `ALLOWED_HOSTS=*`, `CSRF_TRUSTED_ORIGINS=http://localhost:9001,http://127.0.0.1:9001`.
- In Linux è previsto `extra_hosts: host.docker.internal:host-gateway`.

Troubleshooting veloce:
- API non raggiungibile: verifica che l’API esponga `:9000` e che `host.docker.internal` risolva dall’interno del container (Linux: richiede `host-gateway`).
- Errori CSRF/CORS: verifica `CSRF_TRUSTED_ORIGINS` nel frontend e CORS nell’API.

## Esempio Compose Integrato (non usato)

Nota: gli stack restano separati (API, Ollama, Qdrant ciascuno nel proprio progetto). Questo blocco è solo un esempio di compose combinato e non è usato né mantenuto in questo repository.

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - '6333:6333'
  ollama:
    image: ollama/ollama:latest
    ports:
      - '11434:11434'
  api:
    build: .
    command: uvicorn app:app --host 0.0.0.0 --port 9000
    ports:
      - '9000:9000'
    environment:
      - DJANGO_VERIFY_URL=http://django:9001/api/token/verify/
    depends_on:
      - qdrant
      - ollama
  django:
    build: ./mosaico
    command: python manage.py runserver 0.0.0.0:9001
    ports:
      - '9001:9001'
    depends_on:
      - api

networks:
  default:
    external:
      name: i-nest_network
```

## Miglioramenti futuri

* Rate limiting con `fastapi-limiter` o middleware custom
* Autorizzazioni e ruoli utente
* Dashboard admin avanzata
* Deployment su Kubernetes / Gunicorn + Nginx

---

## TEST Rapido
## #############################################
python manage.py runserver 0.0.0.0:9001

Utenze:
pc casaccia: demonstration:mosaico1234
laptop:dimostrazione:mosaico1234
callia: dimostrazione:mosaico1234

## #############################################
*Documentazione generata automaticamente.*
