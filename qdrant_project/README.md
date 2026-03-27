# Qdrant Docker Setup

Questo progetto fornisce una configurazione Docker Compose per eseguire Qdrant, un motore di database vettoriale.

## Prerequisiti

* Docker (versione >= 19.03)
* Docker Compose (versione >= 1.27)

## Struttura del progetto

* `docker-compose.yml`: Definisce il servizio Qdrant e il volume per la persistenza dei dati.

## Configurazione

Non sono richieste configurazioni aggiuntive. Il file `docker-compose.yml` predefinito espone la porta 6333 e monta un volume `qdrant_storage`.

## Esecuzione

Per avviare il servizio in background:

```bash
docker-compose up -d

In alcune vesioni di Linux
docker compose up -d
```

```bash per vedere log
docker logs -f qdrant
```

Per fermare e rimuovere i container e la rete creata:

```bash
docker-compose down
```

## Volume

Il volume Docker `qdrant_storage` conserva i dati tra i riavvii del container.

## Utilizzo

Una volta avviato, Qdrant sarà disponibile su `http://localhost:6333`. Puoi interagire tramite REST API o client come `qdrant-client`.

## Esempio in Python

### Installazione librerie

Installa il client ufficiale di Qdrant:

```bash
pip install qdrant-client
```

### Codice di esempio

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Connessione al servizio Qdrant
client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

# Creazione o ricreazione di una collezione
client.recreate_collection(
    collection_name="my_collection",
    vector_size=3,
    distance="Cosine"
)

# Inserimento di punti vettoriali
points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"metadata": "esempio"}),
    PointStruct(id=2, vector=[0.4, 0.5, 0.6], payload={"metadata": "altro"})
]
client.upsert(collection_name="my_collection", points=points)

# Ricerca vettoriale
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, 0.3],
    limit=5
)
print(results)
```

## Interazioni tramite HTTP / curl

### Creare una collezione

```bash
curl -X PUT "http://localhost:6333/collections/my_collection" \
     -H "Content-Type: application/json" \
     -d '{
           "vectors": {"size": 3, "distance": "Cosine"}
         }'
```

### Inserire punti

```bash
curl -X PUT "http://localhost:6333/collections/my_collection/points" \
     -H "Content-Type: application/json" \
     -d '{
           "points": [
             {"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"metadata": "esempio"}},
             {"id": 2, "vector": [0.4, 0.5, 0.6], "payload": {"metadata": "altro"}}
           ]
         }'
```

### Ricerca vettoriale

```bash
curl -X POST "http://localhost:6333/collections/my_collection/points/search" \
     -H "Content-Type: application/json" \
     -d '{
           "vector": [0.1, 0.2, 0.3],
           "limit": 5
         }'
```

## Risorse utili

* [Documentazione ufficiale Qdrant](https://qdrant.tech/documentation/)
* [Repository Docker Qdrant su Docker Hub](https://hub.docker.com/r/qdrant/qdrant)

## Licenza

Questo progetto è rilasciato con licenza MIT.
