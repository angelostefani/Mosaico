m.PHONY: up local down logs

## Avvia lo stack con Ollama esterno (usa OLLAMA_URL dal .env)
up:
	docker compose up -d --build

## Avvia lo stack con Ollama locale (container Docker) e scarica il modello automaticamente
local:
	OLLAMA_URL=http://ollama:11434/api/generate docker compose --profile local up -d --build

## Ferma tutti i container (incluso il profilo local se attivo)
down:
	docker compose --profile local down

## Mostra i log in tempo reale
logs:
	docker compose --profile local logs -f
