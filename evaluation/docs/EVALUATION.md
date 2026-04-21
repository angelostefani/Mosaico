# Guida alla valutazione RAG con RAGAS

Pipeline: `eval_batch.py` raccoglie le risposte del sistema → `eval_ragas.py` calcola le metriche.

Gli script si trovano in `evaluation/` e sono indipendenti dal codice del servizio `ai_api/`.
Possono puntare a qualsiasi istanza dell'API (locale o remota) tramite `--url`.

---

## Prerequisiti

### 1. Stack Mosaico in esecuzione

```bash
# Dalla root del progetto
make up
# oppure
docker compose up -d --build
```

Verifica che l'API risponda:
```bash
curl http://localhost:9000/healthz
```

### 2. Documenti indicizzati

I documenti su cui verranno poste le domande del dataset devono essere già caricati nella collection corretta tramite l'interfaccia Django (`http://localhost:9001`) o l'endpoint `/upload`.

### 3. Modello LLM disponibile su Callia

```bash
curl http://192.168.118.218:11434/api/tags | python -m json.tool | grep gpt-oss
```

### 4. Dipendenze Python installate

```bash
cd evaluation
pip install -r requirements.txt
```

---

## Struttura cartelle

```
evaluation/
  eval_batch.py          ← Fase 1: raccoglie risposte RAG via /evaluation
  eval_ragas.py          ← Fase 2: calcola metriche RAGAS sul CSV
  requirements.txt       ← dipendenze eval (ragas, langchain-*, ecc.)
  datasets/
    eval_dataset.json        ← 201 Q&A (dataset completo)
    eval_dataset_mini.json   ← 10 Q&A (dataset mini per test rapidi)
  results/               ← output CSV (gitignored)
  docs/
    EVALUATION.md
    rag_evaluation_paper.md
```

---

## Fase 1 — Raccolta risposte RAG (`eval_batch.py`)

```bash
cd evaluation

# API locale (Docker sul laptop)
python eval_batch.py \
  --url http://localhost:9000 \
  --dataset datasets/eval_dataset_mini.json \
  --username enea --collection RECON \
  --model gemma4:e4b \
  --output results/risultati_$(date +%Y%m%d).csv

# API locale (Docker sul laptop)
# POWERSHELL
python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_mini.json `
  --username enea `
  --collection RECON `
  --model gemma4:e4b `
  --output ".\results\risultati_$(Get-Date -Format yyyyMMdd).csv"

# API remota (server)
python eval_batch.py \
  --url http://192.168.118.218:9000 \
  --dataset datasets/eval_dataset_mini.json \
  --username enea --collection RECON \
  --model gpt-oss:20b \
  --output results/risultati_server_$(date +%Y%m%d).csv

# POWERSHELL
# API remota (server)
python .\eval_batch.py `
  --url http://192.168.118.218:9000 `
  --dataset .\datasets\eval_dataset_mini.json `
  --username enea `
  --collection RECON `
  --model gpt-oss:20b `
  --output ".\results\risultati_server_$(Get-Date -Format yyyyMMdd).csv"


**Parametri:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--url` | `http://localhost:9000` | URL dell'API FastAPI |
| `--dataset` | `datasets/eval_dataset.json` | File JSON con le domande |
| `--output` | `eval_results_<timestamp>.csv` | File CSV di output |
| `--username` | — | Tenant (se multi-tenant) |
| `--collection` | — | Nome collection Qdrant |
| `--model` | da `.env` | Override modello LLM |
| `--token` | — | JWT Bearer (se `SKIP_AUTH=false`) |

---

## Fase 2 — Calcolo metriche RAGAS (`eval_ragas.py`)

### Test rapido

```bash
python eval_ragas.py \
  --input results/risultati_$(date +%Y%m%d).csv \
  --metrics context_recall,answer_correctness \
  --model gpt-oss:20b \
  --timeout 900 \
  --batch-size 5 \
  --verbose
```

### Valutazione completa

```bash
python eval_ragas.py \
  --input results/risultati_$(date +%Y%m%d).csv \
  --output results/ragas_$(date +%Y%m%d).csv \
  --model gpt-oss:20b \
  --timeout 900 \
  --verbose
```

# POWERSHELL
python .\eval_ragas.py `
  --input ".\results\risultati_$(Get-Date -Format yyyyMMdd).csv" `
  --output ".\results\ragas_$(Get-Date -Format yyyyMMdd).csv" `
  --model gpt-oss:20b `
  --timeout 900 `
  --verbose


  python .\eval_ragas.py `
  --input ".\results\risultati_$(Get-Date -Format yyyyMMdd).csv" `
  --output ".\results\ragas_server_$(Get-Date -Format yyyyMMdd).csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --model gemma4:e4b `
  --timeout 900 `
  --verbose

**Parametri:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--input` | — | CSV da `eval_batch.py` (**obbligatorio**) |
| `--output` | `ragas_<timestamp>.csv` | CSV di output con le metriche |
| `--ollama-url` | da `.env` | Override `OLLAMA_URL` |
| `--model` | da `.env` | Override modello LLM |
| `--embedding-model` | `all-MiniLM-L6-v2` | Modello embeddings HuggingFace |
| `--metrics` | tutte e 5 | Sottoinsieme metriche (virgola-separato) |
| `--batch-size` | `10` | Campioni per chiamata `evaluate()` |
| `--timeout` | `600` | Secondi per chiamata LLM |
| `--verbose` | off | Stampa punteggi per ogni campione |

Se interrotto con `Ctrl+C`, i risultati parziali vengono salvati automaticamente.

---

## Configurazione attuale (`.env` nella root del progetto)

```
OLLAMA_URL   = http://192.168.118.218:11434/api/generate   # server Callia
OLLAMA_MODEL = gpt-oss:20b
EMBEDDING_MODEL = all-MiniLM-L6-v2
QDRANT_SCORE_THRESHOLD = 0.45
CHAT_CANDIDATES = 80
CHAT_RESULT_LIMIT = 7
MMR_LAMBDA = 0.80
```

---

## Tempi stimati

| Fase | Campioni | Tempo stimato |
|------|----------|---------------|
| `eval_batch.py` dataset mini | 10 | ~2 min |
| `eval_batch.py` dataset completo | 201 | 10–30 min |
| `eval_ragas.py` test rapido | 5 × 2 metriche | 5–15 min |
| `eval_ragas.py` completo | 201 × 5 metriche | 4–8 ore (gpt-oss:20b) |

---

## Troubleshooting

**`Connection refused` su `localhost:9000`** → stack Docker non avviato. Eseguire `make up`.

**`TimeoutError` in RAGAS** → aumentare `--timeout 900` o verificare che il modello sia caricato in VRAM su Callia.

**Tutti i punteggi `NaN`** → il LLM non restituisce JSON valido. Verificare connettività con `curl http://192.168.118.218:11434/api/tags`.

**`ImportError: No module named 'ragas'`** → eseguire `pip install -r requirements.txt` dalla cartella `evaluation/`.

**`n_chunks=0` per molti casi** → i documenti non sono caricati nella collection corretta, oppure `QDRANT_SCORE_THRESHOLD` è troppo alto.
