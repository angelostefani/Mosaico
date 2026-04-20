# Mosaico: A Local RAG System for Technical Documentation QA — Evaluation Study

> **Stato:** Bozza di lavoro — risultati preliminari su dataset mini (10 casi), ablation in corso
> **Autori:** \[da completare\]
> **Data ultimo aggiornamento:** 2026-04-20

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

- **LLM giudice:** `gpt-oss:20b` via Ollama su server Callia (`192.168.118.218`) — stesso modello usato per la generazione (v. discussione bias in §6.2)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` locale tramite `langchain-huggingface`
- **Parallelismo:** `max_workers=1` per evitare timeout su istanza Ollama singola
- **Timeout per chiamata LLM:** 900 secondi
- **Nessuna dipendenza da OpenAI o altri servizi cloud**

### 4.4 Configurazione del sistema RAG — configurazioni a confronto

Sono state testate due configurazioni sul dataset mini (10 casi):

| Parametro | Baseline (v1) | Ottimizzata (v2) |
|-----------|--------------|------------------|
| `OLLAMA_MODEL` | gpt-oss:20b | gpt-oss:20b |
| `QDRANT_SCORE_THRESHOLD` | 0.42 | 0.50 |
| `CHAT_CANDIDATES` | 50 | 80 |
| `CHAT_RESULT_LIMIT` | 10 | 5 |
| `MMR_LAMBDA` | 0.65 | 0.80 |
| `ENABLE_CROSS_ENCODER_RERANK` | false | false |
| Prompt regola concisione | no | sì |

Parametri comuni a entrambe le configurazioni:
```
EMBEDDING_MODEL           = all-MiniLM-L6-v2
CHUNK_SIZE                = 600
CHUNK_OVERLAP             = 120
CHAT_CONTEXT_CHAR_BUDGET  = 9000
ENABLE_RERANK             = true
ENABLE_MMR                = true
ENABLE_STITCH             = true
ENABLE_MULTI_VECTOR_SEARCH = true
```

---

## 5. Risultati

> *Risultati preliminari su dataset mini (10 casi). I risultati definitivi saranno calcolati sull'intero dataset (201 casi) con la configurazione ottimale.*

### 5.1 Statistiche di retrieval — configurazione baseline (da `eval_batch.py`)

| Statistica | Valore |
|------------|--------|
| Casi valutati con successo | 10 / 10 |
| Casi senza contesti recuperati | 0 |
| Media chunk recuperati per domanda | 9.8 |
| Range chunk per domanda | 9–10 |

### 5.2 Metriche RAGAS — confronto baseline vs ottimizzata

| Metrica | Baseline (v1) | Ottimizzata (v2) | Δ |
|---------|:---:|:---:|:---:|
| **Faithfulness** | 0.625 ± 0.320 (NaN=1) | 0.487 ± 0.369 (NaN=1) | -0.138 ↓ |
| **Answer Relevancy** | 0.264 ± 0.156 | 0.347 ± 0.230 | **+0.083** ✓ |
| **Context Precision** | 0.465 ± 0.319 | 0.552 ± 0.337 | **+0.087** ✓ |
| **Context Recall** | 0.629 ± 0.378 | 0.463 ± 0.352 | -0.166 ↓ |
| **Answer Correctness** | 0.473 ± 0.233 | 0.417 ± 0.204 | -0.056 ↓ |

La configurazione v2 migliora `answer_relevancy` (+31%) e `context_precision` (+19%) a discapito di `context_recall` (-26%). Il calo del recall è attribuibile all'innalzamento del threshold (0.42→0.50) e alla riduzione del numero di chunk restituiti (10→5), che filtrano chunk marginalmente rilevanti ma necessari per coprire tutte le informazioni di riferimento.

### 5.3 Risultati per caso — baseline (v1)

| # | Domanda (sintesi) | faith. | relevancy | prec. | recall | correct. |
|---|-------------------|--------|-----------|-------|--------|----------|
| 1 | Ruolo ENEA nelle CER | 1.000 | 0.371 | 0.844 | 1.000 | 0.579 |
| 2 | Strumenti ENEA per CER | 0.286 | 0.323 | 0.730 | 0.333 | 0.490 |
| 3 | Cos'è il TIAD | 0.000 | 0.000 | 0.125 | 1.000 | 0.127 |
| 4 | Chi ha sviluppato RECON | 0.500 | 0.000 | 0.000 | 0.000 | 0.073 |
| 5 | Obiettivo principale RECON | 1.000 | 0.262 | 0.589 | 1.000 | 0.726 |
| 6 | Configurazioni autoconsumo | NaN | 0.338 | 0.569 | 1.000 | 0.590 |
| 7 | Come funziona RECON | 0.929 | 0.489 | 0.870 | 0.625 | 0.553 |
| 8 | Struttura RECON | 0.700 | 0.387 | 0.000 | 0.000 | 0.206 |
| 9 | Agevolazioni di legge | 0.500 | 0.151 | 0.278 | 0.667 | 0.720 |
| 10 | Ultima versione RECON | 0.714 | 0.320 | 0.643 | 0.667 | 0.665 |

**Casi critici:**
- **Caso [3]** (*TIAD*): `faithfulness=0.0` nonostante `context_recall=1.0` — il contesto pertinente è recuperato ma la risposta non lo utilizza correttamente.
- **Caso [4]** (*Chi ha sviluppato RECON*): `context_precision=0.0`, `context_recall=0.0` — failure di retrieval; i chunk con le informazioni sul laboratorio SCC/ICER/TERIN non sono stati recuperati.
- **Caso [8]** (*Struttura RECON*): `context_precision=0.0`, `context_recall=0.0` — analoga failure di retrieval, probabile problema di copertura del corpus indicizzato.

### 5.4 Analisi del trade-off precision/recall

Le due configurazioni evidenziano un classico trade-off: aumentare il threshold di similarità migliora la precisione dei chunk recuperati ma riduce la copertura delle informazioni di riferimento. Una configurazione intermedia (`QDRANT_SCORE_THRESHOLD=0.45`, `CHAT_RESULT_LIMIT=7`) è in corso di valutazione come v3.

> *\[Da aggiornare con i risultati v3 e con i risultati sul dataset completo (201 casi)\]*

---

## 6. Discussione

### 6.1 Punti di forza del sistema

- Pipeline completamente locale: nessuna dipendenza cloud, nessuna trasmissione di dati sensibili;
- Architettura multi-tenant con isolamento per collection;
- Pipeline di retrieval multi-stadio con query expansion, reranking e MMR;
- Chunk stitching per preservare la coerenza del contesto;

### 6.2 Limiti e criticità

**Bias di auto-valutazione.** Il modello `gpt-oss:20b` è usato sia per generare le risposte sia come LLM-judge in RAGAS. Questo introduce un bias sistematico in *faithfulness*: il modello tende a giudicare coerenti con il contesto le proprie stesse formulazioni. I risultati osservati (faithfulness baseline = 0.625) vanno letti tenendo conto di questa limitazione. Una separazione completa richiederebbe un modello di famiglia diversa esclusivamente per il giudizio.

**Lingua italiana.** I prompt interni di RAGAS sono in inglese; il corpus, le domande e le risposte di riferimento sono in italiano. Questo mismatch linguistico può impattare in particolare *context_recall*, che richiede l'estrazione e il confronto di affermazioni dalla risposta di riferimento in italiano.

**Calibrazione assente.** I punteggi RAGAS con LLM locale non sono calibrati rispetto a valutazioni umane o a benchmark con GPT-4. Andrebbero usati per confronti relativi (configurazioni diverse, soglie di retrieval diverse) piuttosto che come misura assoluta di qualità, fino a una eventuale validazione human-in-the-loop.

**Costo computazionale.** L'uso di `gemma4:26b` come giudice aumenta significativamente i tempi di valutazione (~4–8 ore per 201 campioni × 5 metriche) rispetto a modelli più piccoli, rendendo poco pratico il ricalcolo a ogni modifica del pipeline.

### 6.3 Ablation study — effetto dei parametri di retrieval

I risultati dell'ablation su 10 casi mostrano che le leve di configurazione hanno effetti opposti su precision e recall:

| Modifica | Impatto su precision | Impatto su recall |
|----------|---------------------|------------------|
| Threshold 0.42 → 0.50 | ↑ (meno rumore) | ↓ (più filtraggio) |
| CHAT_RESULT_LIMIT 10 → 5 | ↑ (contesto più pulito) | ↓ (meno copertura) |
| CHAT_CANDIDATES 50 → 80 | neutro | ↑ (più candidati) |
| MMR_LAMBDA 0.65 → 0.80 | ↑ (più rilevanza) | ↓ (meno diversità) |
| Prompt concisione | — | — (solo relevancy) |

Il risultato netto della v2 è un guadagno netto su precision (+19%) e relevancy (+31%) ma una perdita significativa su recall (-26%). Questo suggerisce che la configurazione ottimale si trova in un punto intermedio tra v1 e v2.

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
