import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

def main():
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "my_collection"

    # 1) Elimina la collezione se già esiste
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # 2) Crea la collezione (768 dimensioni, distanza coseno)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

    # 3) Genera 5 embedding di esempio
    points = []
    for i in range(5):
        embedding = np.random.rand(768).tolist()
        payload = {"doc_id": i, "text": f"Test chunk {i}"}
        points.append(PointStruct(id=i, vector=embedding, payload=payload))

    # 4) Carica i punti
    client.upload_points(collection_name, points)

    # 5) Esegui la ricerca di similarità con `search(...)`
    query_vector = np.random.rand(768).tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,          # i primi 3 risultati
        with_payload=True # includi il payload nella risposta
    )

    # 6) Stampa i risultati
    print("Risultati della ricerca:")
    for r in results:
        print(f"- ID: {r.id}, Score={r.score:.4f}, Payload={r.payload}")

if __name__ == "__main__":
    main()
