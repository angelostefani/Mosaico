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
    eval_dataset_EN_CER_test.json   ← 50 Q&A in inglese (collection CER-EN)
    eval_dataset_IT_CER_test.json   ← 50 Q&A in italiano (collection CER-IT)
    eval_dataset_EN_CER_raw.json    ← dataset grezzo pre-pulizia
    eval_dataset_EN_CER_clean.json  ← dataset pulito pre-split
    eval_dataset_EN_CER_val.json    ← split di validazione
  results/               ← output CSV (gitignored)
  docs/
    EVALUATION.md
    RAGAS_EXPERIMENTS.md ← piano completo esperimenti e stato avanzamento
```

---

## Fase 1 — Raccolta risposte RAG (`eval_batch.py`)

### PowerShell — esempio dataset EN (API locale)

```powershell
cd evaluation

python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_EN_CER_test.json `
  --username enea --collection CER-EN `
  --output ".\results\batch_EN_<config>.csv"
```

### PowerShell — esempio dataset IT (API locale)

```powershell
python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_IT_CER_test.json `
  --username enea --collection CER-IT `
  --output ".\results\batch_IT_<config>.csv"
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

### Valutazione standard (3 metriche, 50 Q&A)

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_EN_<config>.csv" `
  --output ".\results\ragas_EN_<config>.csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --judge-model qwen3:8b --no-think `
  --metrics faithfulness,context_precision,context_recall `
  --batch-size 5 --timeout 900 --verbose --exclude-suspicious
```

### Valutazione completa (5 metriche — solo baseline e Naive RAG)

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_EN_<config>.csv" `
  --output ".\results\ragas_EN_<config>.csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --judge-model qwen3:8b --no-think `
  --metrics faithfulness,context_precision,context_recall,answer_relevancy,answer_correctness `
  --batch-size 5 --timeout 900 --verbose --exclude-suspicious
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
| `--exclude-suspicious` | off | Esclude righe con context_precision=0 e context_recall=0 (probabile fallimento giudice) |
| `--batch-size` | `10` | Campioni per chiamata `evaluate()` |
| `--timeout` | `600` | Secondi per chiamata LLM |
| `--max-tokens` | `8192` | Token massimi per risposta LLM |
| `--verbose` | off | Stampa punteggi per ogni campione |

Se interrotto con `Ctrl+C`, i risultati parziali vengono salvati automaticamente.

---

## Configurazione baseline (`.env` nella root del progetto)

```dotenv
OLLAMA_URL=http://192.168.118.218:11434/api/generate   # server Callia
OLLAMA_MODEL=gemma4:26b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
QDRANT_SCORE_THRESHOLD=0.20
CHAT_CANDIDATES=100
CHAT_RESULT_LIMIT=10
CHAT_CONTEXT_CHAR_BUDGET=10000
CHAT_HISTORY_CHAR_BUDGET=0
MMR_LAMBDA=0.65
CROSS_ENCODER_TOP_K=30
ENABLE_RERANK=true
ENABLE_MMR=true
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
ENABLE_CROSS_ENCODER_RERANK=true
```

Per l'ablation: modificare i flag `ENABLE_*` e riavviare lo stack con `docker compose down && docker compose up -d --build`.
Vedere [RAGAS_EXPERIMENTS.md](RAGAS_EXPERIMENTS.md) per la tabella completa delle configurazioni.

---

## Tempi stimati (verificati su H100)

| Fase | Campioni | Metriche | Tempo stimato |
|------|----------|----------|---------------|
| `eval_batch.py` dataset EN/IT | 50 | — | ~5 min |
| `eval_ragas.py` 3 metriche | 50 | 3 | ~45 min (qwen3:8b giudice) |
| `eval_ragas.py` 5 metriche | 50 | 5 | ~75 min (qwen3:8b giudice) |
| Ablation completa (6 config) | 50 × 6 | 3–5 | ~4–6 ore |

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
