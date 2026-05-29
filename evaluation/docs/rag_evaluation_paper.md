# Mosaico: A Local RAG System for Technical Documentation QA — Evaluation Study

> **Stato:** Bozza di lavoro — ablation in corso (2/6 config complete)
> **Autori:** \[da completare\]
> **Data ultimo aggiornamento:** 2026-05-29

---

## Abstract

> *\[Da scrivere a risultati definitivi\]*
>
> Proposta: descrivere brevemente il sistema Mosaico, il dominio applicativo (documentazione tecnica ENEA/RECON sulle Comunità Energetiche Rinnovabili), il dataset di valutazione (201 domande con risposta di riferimento), le metriche RAGAS calcolate con un LLM locale, e i risultati principali. Evidenziare l'aspetto "fully local" (nessuna dipendenza cloud) come contributo distintivo.

---

## 1. Introduzione

### 1.1 Motivazione

L'accesso rapido e preciso a documentazione tecnica complessa rappresenta un'esigenza crescente in contesti istituzionali e industriali. I sistemi RAG (*Retrieval-Augmented Generation*) offrono la possibilità di interrogare corpora documentali in linguaggio naturale, ma la loro valutazione rigorosa — specialmente in ambienti *fully local* senza dipendenze da API cloud — rimane un campo aperto.

Questo lavoro presenta **Mosaico**, una piattaforma modulare per il document QA basata su RAG, e ne documenta la metodologia di valutazione applicata al dominio delle **Comunità Energetiche Rinnovabili (CER)** e del simulatore **RECON** sviluppato da ENEA.

### 1.2 Dominio applicativo

Il corpus di riferimento comprende la documentazione tecnica di ENEA relativa a:
- Il simulatore **RECON** (*Renewable Energy Community ecONomic simulator*) per la valutazione energetica, economica e finanziaria delle CER;
- Materiali informativi sulle Comunità Energetiche Rinnovabili (CER) e sui Gruppi di Autoconsumatori Collettivi (GAC);
- Il quadro normativo di riferimento (D.lgs. 199/2021, TIAD/ARERA).

### 1.3 Contributi

- Descrizione di un'architettura RAG *fully local* (LLM, embedding, vector DB) senza dipendenze da servizi cloud;
- Dataset di valutazione di 201 coppie domanda/risposta di riferimento in italiano sul dominio CER/RECON;
- Applicazione del framework **RAGAS** con LLM locale per il calcolo di metriche automatiche di qualità;
- Analisi dell'affidabilità delle metriche RAGAS con LLM locale (gemma4:26b) su corpus in lingua italiana.

---

## 2. Sistema Mosaico

### 2.1 Architettura generale

```
Browser → Django Frontend (:9001) → FastAPI AI API (:9000)
                                          ├→ Qdrant (:6333)   — Vector DB
                                          ├→ Ollama (:11434)  — LLM locale
                                          └→ PostgreSQL (:5433) — Persistenza
```

L'API FastAPI implementa l'intero pipeline RAG ed espone gli endpoint `/chat`, `/chat/stream` e `/evaluation`. L'architettura è completamente locale: non viene effettuata nessuna chiamata a servizi cloud durante l'inferenza.

### 2.2 Ingestion dei documenti

I documenti vengono processati all'upload con il seguente flusso:

1. **Estrazione testo** — supporto per `.pdf` (pdfplumber), `.docx` (python-docx), `.xlsx` (openpyxl), `.txt`, `.json`;
2. **Normalizzazione** — lowercase, rimozione caratteri di controllo Unicode, collasso degli spazi;
3. **Chunking** — finestra scorrevole con sentence-boundary detection; parametri configurabili via `.env`:
   - `CHUNK_SIZE = 600` caratteri
   - `CHUNK_OVERLAP = 120` caratteri
4. **Embedding** — modello `sentence-transformers/all-MiniLM-L6-v2` (384 dimensioni);
5. **Indicizzazione** — vettori e payload (`text`, `source_file`, `page_number`, `chunk_index`) in Qdrant; isolamento multi-tenant tramite collection separate (`{username}_{collection}`).

### 2.3 Pipeline di retrieval

Al momento della query, il pipeline esegue le seguenti fasi:

| Fase | Descrizione | Flag |
|------|-------------|------|
| **Query expansion** | Tokenizzazione della domanda, selezione di fino a 3 termini aggiuntivi per ricerche secondarie | `ENABLE_MULTI_VECTOR_SEARCH` |
| **Vector search** | Ricerca ANN in Qdrant (top-*k* = 50 candidati) con query primaria + espansioni | — |
| **Threshold filtering** | Filtro per score ≥ 0.42 con fallback adattivo (75% → 50% → 0.0) | `QDRANT_SCORE_THRESHOLD` |
| **Reranking** | Combinazione lineare: 60% score vettoriale + 25% similarità testuale + 15% keyword overlap | `ENABLE_RERANK` |
| **Deduplication** | Fingerprint-based + SequenceMatcher (soglia 0.97) | — |
| **MMR** | *Maximal Marginal Relevance* (λ = 0.65) per bilanciare rilevanza e diversità | `ENABLE_MMR` |
| **Chunk stitching** | Unione di chunk consecutivi della stessa fonte; budget 9000 caratteri (~2500 token) | `ENABLE_STITCH` |

### 2.4 Generazione della risposta

Il contesto assemblato viene passato al modello LLM locale tramite Ollama. Il prompt include:
- Istruzioni di sistema (regole RAG, difesa da prompt injection);
- Scope opzionale della collection;
- Contesto recuperato;
- Storico della conversazione (budget 3000 caratteri);
- Domanda dell'utente.

**Modello:** `gemma4:26b` (Ollama, server Callia `192.168.118.218`)
**Timeout:** 180 secondi

---

## 3. Dataset di valutazione

### 3.1 Composizione

Il dataset è composto da **201 coppie domanda/risposta di riferimento** in italiano, costruite manualmente a partire dalla documentazione ufficiale RECON/ENEA.

| Proprietà | Valore |
|-----------|--------|
| Lingua | Italiano |
| Numero di casi | 201 |
| Dominio | CER, RECON, normativa energetica IT |
| Tipo di domande | Fattuali, procedurali, definitorie |
| Fonte delle risposte di riferimento | Documentazione ufficiale ENEA/RECON |

### 3.2 Caratteristiche delle domande

Le domande coprono diversi livelli di specificità:
- **Definitorie**: *"Cos'è il TIAD?"*, *"Chi ha sviluppato RECON?"*
- **Procedurali**: *"Come funziona il simulatore RECON?"*, *"Come si configura una CER in RECON?"*
- **Comparative/analitiche**: domande che richiedono la sintesi di informazioni distribuite su più documenti

### 3.3 Processo di costruzione

> *\[Da descrivere: chi ha redatto le domande, come sono state validate le risposte di riferimento, quanti documenti coprono il corpus\]*

---

## 4. Metodologia di valutazione

### 4.1 Approccio a due fasi

La valutazione è strutturata in due script distinti e componibili:

**Fase 1 — `eval_batch.py`:** esegue il pipeline RAG completo su tutti i 201 casi tramite l'endpoint `/evaluation` dell'API, raccogliendo per ogni caso: risposta generata, chunk recuperati, sorgenti, numero di chunk. Output: CSV (`eval_results_<timestamp>.csv`).

**Fase 2 — `eval_ragas.py`:** legge il CSV prodotto nella Fase 1 e calcola le metriche RAGAS usando il LLM locale e gli embedding locali. Non richiede di ri-eseguire il RAG. Output: CSV arricchito con le metriche (`ragas_<timestamp>.csv`).

### 4.2 Metriche RAGAS

Le metriche calcolate tramite il framework [RAGAS](https://docs.ragas.io) sono:

| Metrica | Cosa misura | Input richiesti |
|---------|-------------|-----------------|
| **Faithfulness** | Frazione delle affermazioni nella risposta verificabili nel contesto recuperato | risposta, contesti |
| **Answer Relevancy** | Quanto la risposta è pertinente alla domanda (generazione di domande sintetiche + similarità embedding) | domanda, risposta |
| **Context Precision** | Precisione dei contesti: quanti dei chunk recuperati sono effettivamente utili per la risposta di riferimento | domanda, contesti, riferimento |
| **Context Recall** | Copertura: quante affermazioni della risposta di riferimento sono rintracciabili nei contesti | contesti, riferimento |
| **Answer Correctness** | Correttezza fattuale della risposta rispetto al riferimento (LLM + similarità semantica) | risposta, riferimento |

### 4.3 Configurazione RAGAS

- **LLM giudice:** `qwen3:8b` con `--no-think` via Ollama su server Callia (`192.168.118.218`) — modello separato dal generatore per ridurre il bias di auto-valutazione
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` locale tramite `langchain-huggingface`
- **Parallelismo:** `max_workers=1` per evitare timeout su istanza Ollama singola
- **Timeout per chiamata LLM:** 900 secondi
- **Nessuna dipendenza da OpenAI o altri servizi cloud**

### 4.4 Configurazione del sistema RAG — baseline

```
OLLAMA_MODEL              = gemma4:26b
CHUNK_SIZE                = 1000
CHUNK_OVERLAP             = 200
EMBEDDING_MODEL           = all-MiniLM-L6-v2
QDRANT_SCORE_THRESHOLD    = 0.20
CHAT_CANDIDATES           = 100
CHAT_RESULT_LIMIT         = 10
CHAT_CONTEXT_CHAR_BUDGET  = 10000
CHAT_HISTORY_CHAR_BUDGET  = 0
MMR_LAMBDA                = 0.65
CROSS_ENCODER_TOP_K       = 30
ENABLE_RERANK             = true
ENABLE_MMR                = true
ENABLE_STITCH             = true
ENABLE_MULTI_VECTOR_SEARCH = true
ENABLE_CROSS_ENCODER_RERANK = true
```

Le varianti testate nell'ablation study (§5) modificano un solo flag/parametro alla volta rispetto a questa configurazione.

---

## 5. Risultati

> *Ablation in corso — 2 configurazioni complete su 6. I risultati delle restanti configurazioni verranno aggiunti al completamento degli esperimenti (v. `RAGAS_EXPERIMENTS.md`).*

**Dataset:** `eval_dataset_EN_CER_test.json` — 50 domande in inglese, collection `CER-EN`  
**Generatore:** `gemma4:26b` | **Giudice RAGAS:** `qwen3:8b` con `--no-think`

### 5.1 Tab:ablation — Feature flag ablation (parziale)

| Config | Faith. | Ctx Prec. | Ctx Rec. | Ans Rel. | Ans Corr. | N |
|--------|:------:|:---------:|:--------:|:--------:|:---------:|:-:|
| **Baseline** (full pipeline) | 0.722 | 0.507 | 0.705 | 0.784 | 0.476 | 50 |
| No MMR (`ENABLE_MMR=false`) | 0.734 | 0.513 | 0.705 | — | — | 45 |
| No MVS | *in corso* | | | | | |
| Hybrid only (no CE) | *in corso* | | | | | |
| No stitch | *in corso* | | | | | |
| Naive RAG (tutti false) | *in corso* | | | | | |

La disabilitazione di MMR produce variazioni minime su tutte e tre le metriche (Δ < 0.01), suggerendo che con il corpus e il dataset EN la diversificazione MMR non contribuisce significativamente alla qualità del retrieval.

### 5.2 Altre tabelle

> *\[Da completare al termine degli esperimenti: tab:chunking, tab:retrieval_k, tab:mmr, tab:cross_k, tab:reranking, tab:italian\]*

---

## 6. Discussione

### 6.1 Punti di forza del sistema

- Pipeline completamente locale: nessuna dipendenza cloud, nessuna trasmissione di dati sensibili;
- Architettura multi-tenant con isolamento per collection;
- Pipeline di retrieval multi-stadio con query expansion, reranking e MMR;
- Chunk stitching per preservare la coerenza del contesto;

### 6.2 Limiti e criticità

**Bias di auto-valutazione.** Nelle valutazioni correnti il generatore è `gemma4:26b` e il giudice RAGAS è `qwen3:8b` — modelli di famiglia diversa. Questo riduce il rischio di bias sistematico rispetto a configurazioni in cui lo stesso modello genera e giudica. Tuttavia una validazione human-in-the-loop su un sottoinsieme rimane necessaria per calibrare i punteggi in termini assoluti.

**Lingua italiana.** I prompt interni di RAGAS sono in inglese; il corpus, le domande e le risposte di riferimento sono in italiano. Questo mismatch linguistico può impattare in particolare *context_recall*, che richiede l'estrazione e il confronto di affermazioni dalla risposta di riferimento in italiano.

**Calibrazione assente.** I punteggi RAGAS con LLM locale non sono calibrati rispetto a valutazioni umane o a benchmark con GPT-4. Andrebbero usati per confronti relativi (configurazioni diverse, soglie di retrieval diverse) piuttosto che come misura assoluta di qualità, fino a una eventuale validazione human-in-the-loop.

**Costo computazionale.** L'uso di `gemma4:26b` come giudice aumenta significativamente i tempi di valutazione (~4–8 ore per 201 campioni × 5 metriche) rispetto a modelli più piccoli, rendendo poco pratico il ricalcolo a ogni modifica del pipeline.

### 6.3 Ablation study — risultati parziali

Con le due configurazioni disponibili (baseline e no MMR su 50 casi EN):

| Config | Faith. | Ctx Prec. | Ctx Rec. |
|--------|:------:|:---------:|:--------:|
| Baseline | 0.722 | 0.507 | 0.705 |
| No MMR | 0.734 | 0.513 | 0.705 |

La disabilitazione di MMR non produce variazioni significative, indicando che il contributo di MMR sulla qualità del retrieval misurata da RAGAS è marginale sul dataset EN. L'analisi completa dell'effetto delle singole componenti è in corso.

> *\[Da aggiornare al completamento dell'ablation: no MVS, hybrid only, no stitch, naive RAG\]*

### 6.4 Confronto con approcci alternativi

> *\[Opzionale: confronto con BM25 puro, RAG senza reranking, RAG con cross-encoder\]*

---

## 7. Conclusioni e lavori futuri

> *\[Da completare\]*

Possibili direzioni future:
- Valutazione con un LLM di famiglia diversa esclusivamente come giudice per eliminare il bias di auto-valutazione;
- Cross-encoder reranking abilitato (`ENABLE_CROSS_ENCODER_RERANK=true`) e confronto metriche;
- Estensione del dataset a domande multi-documento e domande che richiedono ragionamento;
- Valutazione human-in-the-loop su un sottoinsieme dei 201 casi per calibrare i punteggi RAGAS;
- Ablation study sul contributo delle singole componenti del pipeline (query expansion, MMR, stitching).

---

## Riferimenti

> *\[Da completare\]*

- Es-Haghi et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020
- \[Documentazione RECON/ENEA\]
- \[Normativa D.lgs. 199/2021 e TIAD\]

---

## Appendice A — Esempi dal dataset

> *\[Da selezionare: 5-10 esempi rappresentativi con domanda, risposta di riferimento, risposta generata e metriche RAGAS\]*

---

## Appendice B — Configurazione completa del sistema

> *\[Da aggiungere: tabella completa dei parametri `.env` usati nella valutazione\]*

---

*Documento generato nell'ambito del progetto Mosaico — ENEA/RECON RAG evaluation study.*
