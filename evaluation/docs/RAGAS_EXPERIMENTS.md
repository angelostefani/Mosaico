# Esperimenti RAGAS consigliati per il paper

---

## Esperimento 1 — Ablation study (indispensabile)

È il cuore del paper. Testa ogni flag in isolamento partendo dal full pipeline.

| Config | RERANK | MMR | STITCH | MULTI_VEC |
|--------|--------|-----|--------|-----------|
| Full pipeline | ✓ | ✓ | ✓ | ✓ |
| −rerank | ✗ | ✓ | ✓ | ✓ |
| −MMR | ✓ | ✗ | ✓ | ✓ |
| −stitch | ✓ | ✓ | ✗ | ✓ |
| −multi-vector | ✓ | ✓ | ✓ | ✗ |
| Baseline (tutto off) | ✗ | ✗ | ✗ | ✗ |

**6 config × 100 Q&A × ~20 chiamate RAGAS = ~120K chiamate LLM**
Stima: ~2-3 ore su H100 con `gemma4:e4b` generator + `qwen3:8b` judge.

Metriche da riportare: tutte e 5 + latency retrieval.

---

## Esperimento 2 — Model comparison (raccomandato)

Dimostra che il sistema non è tied a un singolo LLM.

| Generator | Pipeline |
|-----------|----------|
| `gemma4:26b` | full |
| `gpt-oss:20b` | full |
| `qwen3:8b` | full |

**3 config × 100 Q&A.** Stima: ~1 ora aggiuntiva.

---

## Esperimento 3 — Hyperparameter sensitivity (opzionale, post-review)

Varia chunk size (400/600/800), MMR λ (0.5/0.65/0.8), score threshold (0.3/0.45/0.6).
Utile ma richiede molte run — da fare dopo l'acceptance.

---

## Strategia pratica consigliata

```
Fase 1 (mini sweep, ~30 min):
  6 config × 30 Q&A  →  verifica che i punteggi siano sensati, niente tutto-NaN

Fase 2 (ablation completo, ~3h):
  6 config × 100 Q&A con gemma4:e4b + qwen3:8b judge

Fase 3 (model comparison, ~1h):
  3 modelli × full pipeline × 100 Q&A
```

**Minimo assoluto per submission:** solo Fase 2. Senza ablation il paper non ha risultati
quantitativi sui 4 contributi dichiarati, e i reviewer lo noteranno.

---

## Comandi (PowerShell, server remoto)

### Fase 1 — raccolta risposte per una singola config

Imposta i flag nel `.env` prima di ogni run, poi:

```powershell
# Esempio: config full pipeline, dataset mini
python .\eval_batch.py `
  --url http://192.168.118.218:9000 `
  --dataset .\datasets\eval_dataset_mini.json `
  --username enea --collection RECON `
  --model gemma4:e4b `
  --output ".\results\batch_full_mini_$(Get-Date -Format yyyyMMdd).csv"
```

### Fase 2 — metriche RAGAS con giudice separato

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_full_mini_$(Get-Date -Format yyyyMMdd).csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --model gemma4:e4b `
  --judge-model qwen3:8b `
  --timeout 900 `
  --batch-size 5 `
  --verbose
```

---

## Variabili `.env` da cambiare per ogni config ablation

I quattro flag si trovano in `.env` nella root del progetto:

```dotenv
ENABLE_RERANK=true
ENABLE_MMR=true
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
```

Dopo ogni modifica riavviare lo stack perché `ai_api` legge il `.env` all'avvio:

```bash
make down && make up
# oppure
docker compose down && docker compose up -d --build
```

Verificare che i flag siano stati recepiti:

```bash
curl http://192.168.118.218:9000/healthz | python -m json.tool
```

### Valori per ogni config

| Config | `ENABLE_RERANK` | `ENABLE_MMR` | `ENABLE_STITCH` | `ENABLE_MULTI_VECTOR_SEARCH` |
|--------|-----------------|--------------|-----------------|------------------------------|
| Full pipeline | true | true | true | true |
| −rerank | **false** | true | true | true |
| −MMR | true | **false** | true | true |
| −stitch | true | true | **false** | true |
| −multi-vector | true | true | true | **false** |
| Baseline | **false** | **false** | **false** | **false** |

### Flusso completo per una singola config (esempio: −MMR)

```dotenv
# .env
ENABLE_RERANK=true
ENABLE_MMR=false        ← unica modifica
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
OLLAMA_MODEL=gemma4:e4b
```

```powershell
# 1. Riavvia lo stack
docker compose down; docker compose up -d --build

# 2. Raccoglie risposte
python .\eval_batch.py `
  --url http://192.168.118.218:9000 `
  --dataset .\datasets\eval_dataset.json `
  --username enea --collection RECON `
  --model gemma4:e4b `
  --output ".\results\batch_no_mmr.csv"

# 3. Calcola metriche RAGAS
python .\eval_ragas.py `
  --input ".\results\batch_no_mmr.csv" `
  --output ".\results\ragas_no_mmr.csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --model gemma4:e4b `
  --judge-model qwen3:8b `
  --timeout 900 --batch-size 5 --verbose
```

Ripetere cambiando il flag e il nome del file di output per ogni riga della tabella.
