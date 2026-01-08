from qdrant_client import QdrantClient

# Connessione a Qdrant
client = QdrantClient(host="localhost", port=6333)

# Lista delle collezioni esistenti
collections = client.get_collections()
print("Collezioni:", collections)

# Recupera (scroll) i punti dalla collezione "documents"
# scroll restituisce una tupla: (points, next_page_offset)
points, next_page_offset = client.scroll(collection_name="documents", limit=50)

print("Punti della collezione 'documents':")
for point in points:
    print(point)

# Se vuoi visualizzare anche il token per la paginazione, lo puoi stampare:
print("Prossimo offset per lo scroll:", next_page_offset)
