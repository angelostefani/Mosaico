# Documentazione progetto Qdrant

Questa cartella raccoglie la documentazione operativa per eseguire e gestire l'istanza Qdrant avviata tramite Docker Compose.

## Contenuti
- Descrizione rapida dello stack
- Prerequisiti e configurazione
- Avvio, arresto e verifica del servizio
- Gestione dati e volumi
- Risoluzione problemi comuni

## Stack e topologia
- Servizio unico `qdrant` basato su `qdrant/qdrant:latest`.
- Porta esposta: `6333` (HTTP/REST e gRPC con prefer_grpc=false).
- Volume `qdrant_storage` montato su `/qdrant/storage` per la persistenza.
- Limite file aperti elevato a 65.535 (ulimits soft/hard) per evitare errori su carichi intensi.

## Prerequisiti
- Docker >= 19.03
- Docker Compose >= 1.27 (oppure il plugin `docker compose` recente)
- Porta host `6333` libera

## Avvio rapido
```bash
docker compose up -d
# in ambienti con docker-compose classico:
docker-compose up -d
```

### Verifica stato
```bash
docker ps --filter "name=qdrant"
curl -s http://localhost:6333/health | jq .
```
Risposta attesa: `{"status":"ok"}`.

## Arresto e pulizia
```bash
docker compose down          # ferma container e rete
docker volume ls | grep qdrant_storage  # verifica volume
```
Il volume **non** viene eliminato da `down` e mantiene i dati. Per rimuoverlo esplicitamente:
```bash
docker volume rm qdrant_project_qdrant_storage
```
Sostituisci il nome del volume con quello mostrato da `docker volume ls` se differente.

## Parametri chiave del `docker-compose.yml`
- `image: qdrant/qdrant:latest`: usa l'ultima release disponibile su Docker Hub.
- `ports: "6333:6333"`: mappa la porta REST/gRPC su host.
- `volumes: qdrant_storage:/qdrant/storage`: persiste i dati vettoriali.
- `restart: unless-stopped`: il container riparte automaticamente dopo reboot/crash.
- `ulimits.nofile (soft/hard 65535)`: previene errori "too many open files".
- `environment.QDRANT__STORAGE__STORAGE_PATH`: percorso dati interno (default).

## Collegamento da un client Python (esempio)
```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
client.recreate_collection("demo", vector_size=3, distance="Cosine")
client.upsert(
    collection_name="demo",
    points=[
        {"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"note": "esempio"}},
        {"id": 2, "vector": [0.4, 0.5, 0.6], "payload": {"note": "altro"}}
    ],
)
```

## Risoluzione problemi
- Porta 6333 occupata: modifica la mappatura in `docker-compose.yml` (es. `16333:6333`) e riavvia.
- Errori di file aperti: verificare che l'host consenta ulimit >= 65535; su Linux usare `ulimit -n`.
- Volume corrotto: spegnere il container, fare backup/rimozione del volume, poi riavviare per rigenerarlo.
- Aggiornamento immagine: `docker compose pull && docker compose up -d` per applicare l'ultima `latest`.

## Note operative
- Conservare i dati sensibili all'interno del volume; gli aggiornamenti dell'immagine non li sovrascrivono.
- Per ambienti di produzione valutare pinning dell'immagine (es. `qdrant/qdrant:1.8.5`) e backup del volume.

## Riferimenti
- Documentazione ufficiale: https://qdrant.tech/documentation/
- API REST: https://qdrant.tech/documentation/interfaces/rest/
