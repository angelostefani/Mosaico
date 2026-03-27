# AI API & Front-End Mosaico

Progetto: frontend Django con autenticazione JWT e interfaccia per upload documenti, chat RAG e gestione collection, che si interfaccia con le API REST FastAPI (ai_api).

## Architettura

| Componente | Host/porta default | Note |
|---|---|---|
| **Frontend Django** (questo repo) | `:9001` | Sessione utente, JWT, UI |
| **API REST FastAPI** (ai_api) | `:9000` | Pipeline RAG, Qdrant, Ollama |
| **Qdrant** | `:6333` | Database vettoriale |
| **Ollama** | `:11434` | LLM serving |

Networking:
- Browser → Frontend (`:9001`).
- Frontend (server-side) → API tramite `API_BASE` (configurabile via `.env`).
- CORS/CSRF: il frontend include `CSRF_TRUSTED_ORIGINS`; l'API deve consentire CORS dal frontend.

## Caratteristiche

- **Django 5.2** frontend con sessione utente, JWT via DRF SimpleJWT (lifetime 60 min), template Bootstrap responsive.
- **Whitenoise** per la gestione dei file statici.
- **Endpoint UI**:

  | URL | Vista | Auth |
  |---|---|---|
  | `/` | Home | login required |
  | `/login/` | Login + JWT | — |
  | `/register/` | Registrazione | — |
  | `/logout/` | Logout | — |
  | `/upload/` | Upload documenti | login required |
  | `/uploads/` | Storico upload | login required |
  | `/chat/` | Chat RAG autenticata | login required |
  | `/collection-config/` | Configurazione collection | login required |
  | `/public-chat/` | Chat pubblica (no login) | — |
  | `/password/change/` | Cambio password | login required |

- **Sincronizzazione collection** tra pagine tramite `localStorage`.
- **Chat pubblica** accessibile via `?username=<utente>&collection=<collection>` senza autenticazione.
- **Logging** su console e file con rotazione (`logs/mosaico.log`).

## Prerequisiti

- Python 3.11+
- Docker & Docker Compose (per deploy containerizzato)
- Virtualenv consigliato

## Setup (sviluppo locale)

1. Clona il repository.
2. Crea e attiva un virtualenv:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/Mac
   .\.venv\Scripts\activate         # Windows
   ```

3. Installa dipendenze:

   ```bash
   pip install -r requirements.txt
   ```

4. Copia il file di configurazione e personalizzalo:

   ```bash
   cp .env.example .env
   ```

5. Migrazioni e creazione superuser:

   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. Avvia il server:

   ```bash
   python manage.py runserver 0.0.0.0:9001
   ```

## Variabili d'ambiente

Tutte le variabili sono opzionali e hanno valori di default. Configurarle in `.env`.

| Variabile | Default | Descrizione |
|---|---|---|
| `DJANGO_SECRET_KEY` | *(insecure)* | Chiave segreta Django — **obbligatoria in produzione** |
| `DJANGO_DEBUG` | `true` | Modalità debug (`true`/`false`) |
| `ALLOWED_HOSTS` | `*` | Host autorizzati, separati da virgola |
| `CSRF_TRUSTED_ORIGINS` | — | URL trusted per CSRF (es. `http://localhost:9001`) |
| `API_BASE` | `http://192.168.118.218:9000` | URL base dell'API FastAPI |
| `FAKE_TOKEN` | `dev-token` | Token di fallback per dev (sostituisce JWT mancante) |
| `DJANGO_ENABLE_COOP` | `false` | Abilita header `Cross-Origin-Opener-Policy` |
| `LOG_LEVEL` | `INFO` | Livello di log (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_DIR` | `logs` | Directory dove scrivere i log |
| `LOG_FILE` | `mosaico.log` | Nome del file di log |
| `LOG_MAX_BYTES` | `10485760` (10 MB) | Dimensione massima per rotazione log |
| `LOG_BACKUP_COUNT` | `5` | Numero di file di backup log |
| `LOG_SQL` | `false` | Abilita log query SQL Django |

## Endpoint JWT Django

| Metodo | URL | Descrizione |
|---|---|---|
| POST | `/api/token/` | Ottieni `access` & `refresh` token |
| POST | `/api/token/refresh/` | Rinnova access token |
| POST | `/api/token/verify/` | Verifica token |

## Docker

Il frontend gira in un container separato. API, Ollama e Qdrant restano nei loro progetti/compose.

**Prerequisiti**: avvia prima i container di API (porta `9000`), Qdrant e Ollama.

```bash
# Build e avvio
docker compose up -d --build

# Apri il browser su http://localhost:9001
```

Il `docker-compose.yml` monta `./db.sqlite3` e `./logs` come volumi, quindi i dati e i log persistono tra i restart.

### Variabili nel compose

Il compose sovrascrive alcune variabili `.env`:

```
DJANGO_DEBUG=false
API_BASE=http://192.168.153.248:9000
ALLOWED_HOSTS=localhost,127.0.0.1,...
CSRF_TRUSTED_ORIGINS=http://localhost:9001,...
```

Modifica `docker-compose.yml` per adattarli al tuo ambiente.

> **Linux**: `extra_hosts: host.docker.internal:host-gateway` è già incluso per la risoluzione dell'host.

## Log

```bash
# Log applicazione in tempo reale
tail -f logs/mosaico.log

# Log container Docker
docker compose logs -f frontend
```

## Troubleshooting

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `502 Bad Gateway` su `/upload/` | Upload lento, timeout 300s superato | Verificare disponibilità API; aumentare `timeout` in `views.py` |
| API non raggiungibile | `API_BASE` errato o API non avviata | Verificare `API_BASE` nel `.env` e lo stato del container API |
| Errori CSRF | `CSRF_TRUSTED_ORIGINS` mancante | Aggiungere l'URL del frontend in `CSRF_TRUSTED_ORIGINS` |
| JWT non ottenuto al login | API non risponde su `/api/token/` | Controllare che l'API sia avviata; `FAKE_TOKEN` è il fallback |
| Static files non serviti | `collectstatic` non eseguito | Eseguire `python manage.py collectstatic` |

## Struttura del progetto

```
mosaico/
├── mosaico/           # Configurazione Django (settings, urls, wsgi, asgi)
├── ui/                # App Django principale
│   ├── templates/     # Template HTML (Bootstrap)
│   ├── static/        # File statici
│   ├── views.py       # Logica viste
│   ├── forms.py       # Form upload e chat
│   └── urls.py        # Routing URL
├── logs/              # Log applicazione (gitignored)
├── .env.example       # Template variabili d'ambiente
├── docker-compose.yml # Deploy containerizzato
├── Dockerfile         # Immagine frontend
└── docs/              # Documentazione dettagliata
```

## Documentazione

- [docs/flusso_applicativo_mosaico.md](docs/flusso_applicativo_mosaico.md) — architettura, flussi end-to-end, deployment
- [docs/mosaico_documentazione.md](docs/mosaico_documentazione.md) — traccia schermate per presentazione

---

*Documentazione aggiornata al 2026-03-16.*
