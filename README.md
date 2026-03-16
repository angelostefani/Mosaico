<p align="center">
  <img src="./assets/mosaico-banner.svg" alt="Banner Mosaico" width="100%" />
</p>

<h1 align="center">Mosaico</h1>

<p align="center">
  Stack locale per <strong>Document QA</strong> e <strong>chat RAG</strong> con
  <code>FastAPI</code>, <code>Qdrant</code>, <code>Ollama</code> e <code>Django</code>.
</p>

<p align="center">
  <a href="./mosaico_api/README.md">API</a> •
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
| [`mosaico_api/`](./mosaico_api/README.md) | API FastAPI per upload, indicizzazione, chat RAG, JWT e multi-tenant | `9000` |
| [`mosaico/`](./mosaico/README.md) | Frontend Django per autenticazione, upload, chat e gestione collection | `9001` |
| [`qdrant_project/`](./qdrant_project/README.md) | Stack Docker minimale per Qdrant | `6333` |
| [`ollama_project/`](./ollama_project/README.md) | Stack Docker per Ollama e modelli locali | `11434` |
| [`postgres_project/`](./postgres_project/docker-compose.yml) | Stack Docker per PostgreSQL opzionale | `5433 -> 5432` |

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

### 1. Prerequisiti

- Docker e Docker Compose
- Python 3.11+ se vuoi eseguire API o frontend fuori da container

### 2. Avvia i servizi di base

```bash
cd qdrant_project
docker compose up -d
```

```bash
cd ollama_project
docker compose up -d
docker compose exec ollama ollama pull gemma3:1b
```

```bash
cd postgres_project
docker compose up -d
```

> PostgreSQL e opzionale: il backend supporta anche SQLite per lo sviluppo locale.

### 3. Avvia backend e frontend

Per il backend consulta [`mosaico_api/README.md`](./mosaico_api/README.md): contiene setup Python, variabili `.env`, endpoint e note operative.

Per il frontend consulta [`mosaico/README.md`](./mosaico/README.md): contiene setup Django, login/JWT, UI e avvio con Docker o ambiente locale.

## Endpoint principali

| Servizio | URL |
| --- | --- |
| API FastAPI | `http://localhost:9000` |
| Swagger API | `http://localhost:9000/docs` |
| Frontend Django | `http://localhost:9001` |
| Qdrant | `http://localhost:6333` |
| Ollama | `http://localhost:11434` |
| PostgreSQL | `localhost:5433` |

## Perche questa struttura

- Isola UI, API e infrastruttura in moduli separati.
- Permette di usare SQLite in sviluppo e PostgreSQL quando serve persistenza piu robusta.
- Mantiene locale l'intera catena RAG, incluse embedding, retrieval e generazione.
- Riduce l'accoppiamento tra componenti e rende piu semplice il deploy per ambienti diversi.

## Documentazione

- [`mosaico_api/README.md`](./mosaico_api/README.md) per API, configurazione, database, endpoint e testing.
- [`mosaico/README.md`](./mosaico/README.md) per frontend, autenticazione e interfaccia.
- [`qdrant_project/README.md`](./qdrant_project/README.md) per il servizio vettoriale.
- [`ollama_project/README.md`](./ollama_project/README.md) per il serving dei modelli locali.

## Stato del progetto

La documentazione root e stata riallineata ai nomi reali presenti in repository: `mosaico_api/` e `mosaico/`.
