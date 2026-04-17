# Postgres Project Documentation

## Panoramica
Ambiente Docker Compose che avvia un database PostgreSQL 16 preconfigurato. Espone la porta host `5432` e persiste i dati in un volume Docker locale.

## Requisiti
- Docker e Docker Compose installati
- Porta host 5432 libera

## Avvio rapido
```powershell
# da questa cartella
docker compose up -d
```
Il servizio effettua un healthcheck con `pg_isready` ogni 10s.

## Connessione
- Host: `localhost`
- Porta: `5432`
- DB: `app_db`
- User: `postgres`
- Password: `postgres`

## Volume dati
I dati sono conservati nel volume Docker `postgres_data` montato in `/var/lib/postgresql/data`. Per ispezionare:
```powershell
docker volume inspect postgres_data
```

## Spegnimento e pulizia
```powershell
docker compose down           # ferma i container
# per rimuovere anche i dati
# docker compose down -v
```
