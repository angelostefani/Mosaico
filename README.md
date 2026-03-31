<p align="center">
  <img src="./assets/mosaico-banner.svg" alt="Banner Mosaico" width="100%" />
</p>

<h1 align="center">Mosaico</h1>

<p align="center">
  Stack locale per <strong>Document QA</strong> e <strong>chat RAG</strong> con
  <code>FastAPI</code>, <code>Qdrant</code>, <code>Ollama</code> e <code>Django</code>.
</p>

<p align="center">
  <a href="./ai_api/README.md">API</a> •
  <a href="./mosaico/README.md">Frontend</a> •
  <a href="./qdrant_project/README.md">Qdrant</a> •
  <a href="./ollama_project/README.md">Ollama</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-RAG%20API-0F766E?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI badge" />
  <img src="https://img.shields.io/badge/Django-Frontend-1B4332?style=for-the-badge&logo=django&logoColor=white" alt="Django badge" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant badge" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-111827?style=for-the-badge" alt="Ollama badge" />
</p>

## Panoramica

Mosaico e una piattaforma modulare per interrogare documenti in locale tramite una pipeline RAG completa: upload dei file, estrazione e chunking del testo, indicizzazione vettoriale su Qdrant e generazione delle risposte con Ollama.

La repository separa chiaramente i componenti applicativi dai servizi di infrastruttura, cosi puoi avviare, evolvere o sostituire ogni blocco in modo indipendente.

## Cosa include

| Modulo | Ruolo | Porta predefinita |
| --- | --- | --- |
| [`ai_api/`](./ai_api/README.md) | API FastAPI per upload, indicizzazione, chat RAG, JWT e multi-tenant | `9000` |
| [`mosaico/`](./mosaico/README.md) | Frontend Django per autenticazione, upload, chat e gestione collection | `9001` |
| [`qdrant_project/`](./qdrant_project/README.md) | Stack Docker minimale per Qdrant | `6333` |
| [`ollama_project/`](./ollama_project/README.md) | Stack Docker per Ollama e modelli locali | `11434` |
| [`postgres_project/`](./postgres_project/docker-compose.yml) | Stack Docker per PostgreSQL opzionale | `5433 -> 5432` |

## Funzionalita principali

- **Chat RAG con streaming** — le risposte appaiono token per token tramite SSE, eliminando l'attesa
- **Upload multipli** — seleziona piu file contemporaneamente con barra di progresso per ciascuno
- **Formati supportati** — `.txt`, `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.json`
- **Gestione conversazioni** — elimina o esporta in Markdown qualsiasi conversazione dalla sidebar
- **Auto-pull modello Ollama** — con Ollama locale, il modello configurato viene scaricato automaticamente al primo avvio
- **Multi-tenant** — isolamento per `username` e `collection` tramite namespace Qdrant
- **Database duale** — SQLite in sviluppo, PostgreSQL in produzione

## Architettura

```text
Browser
  |
  v
Frontend Django (:9001)
  |
  v
API FastAPI (:9000)
  |--------------------> Qdrant (:6333)
  |--------------------> Ollama (:11434)
  |
  +--------------------> PostgreSQL (:5433, opzionale)
```

## Quick Start

### Prerequisiti

- Docker e Docker Compose
- Python 3.11+ se vuoi eseguire API o frontend fuori da container

### Avvio completo con un solo comando (consigliato)

Il `docker-compose.yml` nella root avvia tutti i servizi insieme: PostgreSQL, Qdrant, API e Frontend.

```bash
# Copia e configura le variabili d'ambiente
cp .env.example .env   # poi edita .env con i tuoi valori
```

| Scenario | Windows (PowerShell) | Linux / macOS | Ollama usato |
|---|---|---|---|
| Ollama su server esterno | `.\start.ps1 up` | `make up` | `OLLAMA_URL` dal `.env` |
| Ollama locale Docker | `.\start.ps1 local` | `make local` | container `ollama` interno |

Con **Ollama locale** il modello configurato in `OLLAMA_MODEL` viene scaricato automaticamente al primo avvio tramite il servizio `ollama-pull`. I riavvii successivi non riscaricano il modello grazie al volume `ollama_data`.

Per fermare tutti i container: `.\start.ps1 down` (Windows) / `make down` (Linux/macOS)

### Avvio per moduli (alternativa)

Se preferisci avviare i componenti separatamente o vuoi piu controllo:

```bash
cd qdrant_project && docker compose up -d
```

```bash
cd ollama_project && docker compose up -d
docker compose exec ollama ollama pull gemma3:1b   # scarica il modello manualmente
```

```bash
cd postgres_project && docker compose up -d
```

Per backend e frontend consulta [`ai_api/README.md`](./ai_api/README.md) e [`mosaico/README.md`](./mosaico/README.md).

## Endpoint principali

| Servizio | URL |
| --- | --- |
| API FastAPI | `http://localhost:9000` |
| Swagger API | `http://localhost:9000/docs` |
| Frontend Django | `http://localhost:9001` |
| Qdrant | `http://localhost:6333` |
| Ollama | `http://localhost:11434` |
| PostgreSQL | `localhost:5433` |

## Endpoint API notevoli

| Metodo | Path | Descrizione |
| --- | --- | --- |
| `POST` | `/upload` | Carica e indicizza un documento |
| `POST` | `/chat` | Chat RAG (risposta completa JSON) |
| `POST` | `/chat/stream` | Chat RAG con streaming SSE (token progressivi) |
| `GET` | `/uploads` | Cronologia upload |
| `GET` | `/conversations` | Elenco conversazioni |
| `DELETE` | `/conversations/{id}` | Elimina una conversazione |
| `GET/PUT/DELETE` | `/collection/config` | Configurazione collection |
| `GET` | `/healthz` | Health check completo |
| `GET` | `/docs` | Swagger UI |

## Perche questa struttura

- Isola UI, API e infrastruttura in moduli separati.
- Permette di usare SQLite in sviluppo e PostgreSQL quando serve persistenza piu robusta.
- Mantiene locale l'intera catena RAG, incluse embedding, retrieval e generazione.
- Riduce l'accoppiamento tra componenti e rende piu semplice il deploy per ambienti diversi.

## Documentazione

- [`ai_api/README.md`](./ai_api/README.md) per API, configurazione, database, endpoint e testing.
- [`mosaico/README.md`](./mosaico/README.md) per frontend, autenticazione e interfaccia.
- [`qdrant_project/README.md`](./qdrant_project/README.md) per il servizio vettoriale.
- [`ollama_project/README.md`](./ollama_project/README.md) per il serving dei modelli locali.

## Stato del progetto

Il modulo API e stato rinominato da `mosaico_api/` a `ai_api/`. Tutta la documentazione e i link sono stati aggiornati di conseguenza.

Funzionalita aggiunte nella revisione corrente: streaming SSE su `/chat/stream`, upload multipli con progress per-file, eliminazione ed esportazione conversazioni, supporto file `.json`, pull automatico del modello Ollama al primo avvio, script `start.ps1` per Windows.
