# Configurazione Docker di Ollama con gemma3:1b (con volume Docker)

Questo progetto fornisce una configurazione Docker per eseguire Ollama con il modello `gemma3:1b`, utilizzando un volume Docker gestito per salvare i modelli in modo persistente.

## Sommario

- [Prerequisiti](#prerequisiti)
- [Struttura del progetto](#struttura-del-progetto)
- [Setup](#setup)
  - [1. Avviare Docker Compose](#1-avviare-docker-compose)
  - [2. Scaricare il modello](#2-scaricare-il-modello)
- [API REST](#api-rest)
  - [Esempio di chiamata](#esempio-di-chiamata)
- [Backup del volume](#backup-del-volume)
- [Accesso da rete locale](#accesso-da-rete-locale)
- [Deploy su server remoto](#deploy-su-server-remoto)
- [Licenza](#licenza)

## Prerequisiti

- Docker Desktop o Docker Engine + Docker Compose
- Connessione Internet per il download del modello

## Struttura del progetto

```plaintext
ollama_project/
├── docker-compose.yml
```

## Setup

### 1. Avviare Docker Compose

Assicurati di trovarti nella cartella contenente `docker-compose.yml` e avvia:

```bash
docker compose up -d
```

Questo comando:
- Avvia il container `ollama`
- Crea un volume Docker chiamato `ollama_data`
- Avvia il server `ollama serve` in ascolto sulla porta 11434

```bash Verifica che il container sia UP
docker ps
```

### 2. Scaricare il modello

Per scaricare manualmente `gemma3:1b` nel volume, esegui:

```bash
docker compose exec -d ollama ollama pull gemma3:1b
```

```bash per controllare lo stato
docker exec -it ollama ollama list
```

```bash per vedere i log
docker logs -f ollama
```

## API REST

Ollama espone un'API sulla porta `11434`.

### Esempio di chiamata

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:1b",
    "messages": [
      {"role": "user", "content": "Ciao, sei attivo?"}
    ]
  }'
```

## Backup del volume

Per eseguire un backup del volume `ollama_data`:

```bash
docker run --rm -v ollama_data:/data -v $(pwd):/backup busybox tar czf /backup/ollama_data_backup.tar.gz /data
```

Per ripristinare:

```bash
docker run --rm -v ollama_data:/data -v $(pwd):/backup busybox tar xzf /backup/ollama_data_backup.tar.gz -C /
```

## Accesso da rete locale

Se il tuo computer ha IP `192.168.1.10`, puoi accedere a Ollama da altri dispositivi della rete con:

```
http://192.168.1.10:11434
```

Assicurati che:
- Il firewall permetta connessioni sulla porta `11434`
- Docker stia esponendo il servizio su `0.0.0.0` (già impostato)

## Deploy su server remoto

Puoi usare qualsiasi VPS con Docker installato. Passaggi:

1. Copia i file nel server con `scp`:
   ```bash
   scp -r ollama_project user@your-server:/home/user/
   ```

2. Accedi al server:
   ```bash
   ssh user@your-server
   cd /home/user/ollama_project
   ```

3. Avvia Ollama:
   ```bash
   docker compose up -d
   docker compose exec ollama ollama pull gemma3:1b
   ```

4. Accedi da remoto usando l’IP pubblico del server:
   ```
   http://<IP_DEL_SERVER>:11434
   ```

## Licenza

Questo progetto è distribuito con licenza MIT. Puoi modificarlo e riutilizzarlo liberamente.
