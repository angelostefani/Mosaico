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

### 3. Modelli LLM disponibili su Callia

```bash
curl http://192.168.118.218:11434/api/tags | python -m json.tool
```

Modelli confermati disponibili:

| Modello | Ruolo |
|---------|-------|
| `gemma4:e4b` | generatore (ablation, sweep rapidi) |
| `gemma4:26b` | generatore (model comparison) |
| `gpt-oss:20b` | generatore alternativo |
| `qwen3:8b` | giudice RAGAS (separato dal generatore) |

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
    eval_dataset.json        ← 100 Q&A (dataset completo)
    eval_dataset_mini.json   ← 10 Q&A (dataset mini per test rapidi)
  results/               ← output CSV (gitignored)
  docs/
    EVALUATION.md
    RAGAS_EXPERIMENTS.md
```

---

## Fase 1 — Raccolta risposte RAG (`eval_batch.py`)

### PowerShell — API locale (Docker sul laptop)

```powershell
cd evaluation

python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_mini.json `
  --username enea --collection RECON `
  --model gemma4:e4b `
  --output ".\results\batch_mini_$(Get-Date -Format yyyyMMdd).csv"
```

### PowerShell — API remota (server Callia)

```powershell
python .\eval_batch.py `
  --url http://192.168.118.218:9000 `
  --dataset .\datasets\eval_dataset.json `
  --username enea --collection RECON `
  --model gemma4:e4b `
  --output ".\results\batch_full_$(Get-Date -Format yyyyMMdd).csv"
```

**Parametri:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--url` | `http://localhost:9000` | URL dell'API FastAPI |
| `--dataset` | `datasets/eval_dataset.json` | File JSON con le domande |
| `--output` | `eval_results_<timestamp>.csv` | File CSV di output |
| `--username` | — | Tenant (obbligatorio se multi-tenant) |
| `--collection` | — | Nome collection Qdrant |
| `--model` | da `.env` | Override modello generatore LLM |
| `--token` | — | JWT Bearer (se `SKIP_AUTH=false`) |

---

## Fase 2 — Calcolo metriche RAGAS (`eval_ragas.py`)

### Test di validazione setup (2 metriche, 10 campioni)

Eseguire prima di ogni campagna di valutazione per verificare che `qwen3:8b` produca
punteggi validi (NaN=0):

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_mini_$(Get-Date -Format yyyyMMdd).csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --model gemma4:e4b `
  --judge-model qwen3:8b `
  --no-think `
  --metrics faithfulness,context_precision `
  --timeout 900 --batch-size 5 --verbose
```

### Valutazione completa (5 metriche, dataset 100 Q&A)

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_full_$(Get-Date -Format yyyyMMdd).csv" `
  --output ".\results\ragas_full_$(Get-Date -Format yyyyMMdd).csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --model gemma4:e4b `
  --judge-model qwen3:8b `
  --no-think `
  --timeout 900 --batch-size 5 --verbose
```

**Parametri:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--input` | — | CSV da `eval_batch.py` (**obbligatorio**) |
| `--output` | `ragas_<timestamp>.csv` | CSV di output con le metriche |
| `--ollama-url` | da `.env` | Override `OLLAMA_URL` |
| `--model` | da `.env` | Modello generatore (usato per `answer_relevancy`) |
| `--judge-model` | uguale a `--model` | Modello giudice per le metriche LLM-based |
| `--no-think` | off | Disabilita thinking mode per il giudice (raccomandato per `qwen3:8b`) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Modello embeddings HuggingFace |
| `--metrics` | tutte e 5 | Sottoinsieme metriche (virgola-separato) |
| `--batch-size` | `10` | Campioni per chiamata `evaluate()` |
| `--timeout` | `600` | Secondi per chiamata LLM |
| `--max-tokens` | `8192` | Token massimi per risposta LLM |
| `--verbose` | off | Stampa punteggi per ogni campione |

Se interrotto con `Ctrl+C`, i risultati parziali vengono salvati automaticamente.

---

## Configurazione attuale (`.env` nella root del progetto)

```dotenv
OLLAMA_URL=http://192.168.118.218:11434/api/generate   # server Callia
OLLAMA_MODEL=gemma4:26b
EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_SCORE_THRESHOLD=0.45
CHAT_CANDIDATES=80
CHAT_RESULT_LIMIT=7
MMR_LAMBDA=0.80
ENABLE_RERANK=true
ENABLE_MMR=true
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
```

Per l'ablation: modificare i flag `ENABLE_*` e riavviare lo stack con `docker compose down && docker compose up -d --build`.
Vedere [RAGAS_EXPERIMENTS.md](RAGAS_EXPERIMENTS.md) per la tabella completa delle configurazioni.

---

## Tempi stimati (verificati su H100)

| Fase | Campioni | Metriche | Tempo stimato |
|------|----------|----------|---------------|
| `eval_batch.py` dataset mini | 10 | — | ~2 min |
| `eval_batch.py` dataset completo | 100 | — | ~10 min |
| `eval_ragas.py` validazione setup | 10 | 2 | ~9 min |
| `eval_ragas.py` completo | 100 | 5 | ~90 min (gemma4:e4b + qwen3:8b) |
| Ablation completa (6 config) | 100 × 6 | 5 | ~9 ore |

---

## Troubleshooting

**`Connection refused` su `localhost:9000`** → stack Docker non avviato. Eseguire `make up`.

**`TimeoutError` in RAGAS** → aumentare `--timeout 900` o verificare che il modello sia caricato in VRAM su Callia.

**Tutti i punteggi `NaN`** → il giudice LLM non restituisce JSON valido.
- Verificare connettività: `curl http://192.168.118.218:11434/api/tags`
- Se si usa `qwen3:8b`, aggiungere `--no-think` per disabilitare il thinking mode

**`ImportError: No module named 'ragas'`** → eseguire `pip install -r requirements.txt` dalla cartella `evaluation/`.

**`n_chunks=0` per molti casi** → i documenti non sono caricati nella collection corretta, oppure `QDRANT_SCORE_THRESHOLD` è troppo alto.

**Punteggi `context_precision=0` e `context_recall=0` con `faithfulness>0.3`** → segnalato come `suspicious_zero` nel CSV — probabile fallimento del giudice LLM su quella domanda specifica.
