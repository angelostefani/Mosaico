# Documentazione progetto Ollama + gemma3:1b

## Panoramica
Ambiente Docker Compose per eseguire Ollama con il modello `gemma3:1b`, usando un volume Docker (`ollama_data`) per la persistenza dei modelli e le API REST in ascolto su `11434`.

## Requisiti
- Docker Desktop (Windows/macOS) oppure Docker Engine + Docker Compose (Linux)
- Connessione internet per il pull iniziale del modello
- Porta `11434` libera sulla macchina host

## Struttura del repository
- `docker-compose.yml`: definisce il servizio `ollama`, il volume `ollama_data` e l'esposizione della porta 11434.
- `README.md`: guida rapida e note operative.
- `docs/`: documentazione aggiuntiva (questo file).

## Avvio rapido
1. Posizionati nella radice del progetto: `docker compose up -d`
2. Scarica il modello: `docker compose exec -d ollama ollama pull gemma3:1b`
3. Verifica stato container: `docker ps`
4. Test API chat:
   ```bash
   curl http://localhost:11434/api/chat \
     -H "Content-Type: application/json" \
     -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Ciao, sei attivo?"}]}'
   ```

## Gestione del modello
- Lista modelli: `docker exec -it ollama ollama list`
- Log servizio: `docker logs -f ollama`
- Aggiornare un modello: ripetere `ollama pull <modello>` dentro il container.

## Backup e ripristino volume
Backup del volume `ollama_data` nella cartella corrente:
```bash
docker run --rm -v ollama_data:/data -v $(pwd):/backup busybox tar czf /backup/ollama_data_backup.tar.gz /data
```
Ripristino da backup:
```bash
docker run --rm -v ollama_data:/data -v $(pwd):/backup busybox tar xzf /backup/ollama_data_backup.tar.gz -C /
```

## Accesso da rete locale
API raggiungibili su `http://<IP_HOST>:11434` (es. `http://192.168.1.10:11434`). Assicurarsi che il firewall consenta la porta 11434 e che Docker esponga su `0.0.0.0` (già previsto nello stack).

## Deploy su server remoto (VPS)
1. Copia file: `scp -r ollama_project user@server:/home/user/`
2. Sul server: `cd /home/user/ollama_project && docker compose up -d`
3. Scarica modello: `docker compose exec ollama ollama pull gemma3:1b`
4. Accedi via `http://<IP_PUBBLICO>:11434`

## Note e licenza
- Licenza: MIT (vedi `README.md`).
- Personalizzazioni: per usare altri modelli sostituire `gemma3:1b` nei comandi di pull e nelle chiamate API.
