FROM python:3.12-slim

# Evita cache pip e migliora log
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dipendenze di sistema minime (curl per healthcheck opzionali)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Installa dipendenze Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia il codice
COPY . /app

# Assicura la directory uploads esista (montabile come volume)
RUN mkdir -p /app/uploads

EXPOSE 9000

# Comando di default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
