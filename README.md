# 🛡️ LLM Prompt Firewall with RAG based Detection

Two-stage adversarial prompt detection: **DistilBERT classifier** + **RAG retrieval** over known attack patterns.

---

## Architecture

```
User Prompt
    │
    ▼
┌────────────────────────────────────────────────┐
│  STAGE 1 — DistilBERT Classifier               │  < 50ms
│  Output: SAFE / JAILBREAK / INJECTION / PII    │
│  + confidence score                            │
└──────────┬─────────────────────────────────────┘
           │
     ┌─────┴─────┐
     │           │
 conf ≥ 0.90   conf < 0.90
     │           │
     │           ▼
     │  ┌──────────────────────────────────────┐
     │  │  STAGE 2 — RAG Verifier              │
     │  │  ChromaDB + sentence-transformers    │
     │  │  → top-k similar known attacks       │
     │  └──────────────────────────────────────┘
     │           │
     └─────┬─────┘
           ▼
      ALLOW / BLOCK
      + explanation
      + similar attacks
           │
           ▼
    Downstream LLM
    (only if ALLOWED)
```

---

## What's Included

| Module | Description |
|---|---|
| `src/data_loader.py` | Downloads HackAPrompt, Alpaca, AdvBench + synthetic fallback |
| `src/preprocessor.py` | Clean, balance classes, stratified split |
| `src/classifier.py` | Fine-tune DistilBERT, inference |
| `src/rag_pipeline.py` | ChromaDB vector store, cosine retrieval, two-stage classify |
| `src/evaluator.py` | Baseline, confusion matrix, latency, FP analysis, multilingual, ablation, token importance |
| `app/streamlit_app.py` | Three-tab demo: Firewall / Analytics / Explainability |
| `app/api.py` | FastAPI REST backend |

---

## Project Structure

```
llm-firewall/
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── classifier.py
│   ├── rag_pipeline.py
│   └── evaluator.py
├── app/
│   ├── streamlit_app.py
│   └── api.py
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_train.py
│   ├── 03_build_vectorstore.py
│   └── 04_evaluate.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── distilbert-firewall/
├── vectorstore/
└── results/
```
---
## Prerequisite
```bash
Install Python
Install Pip
```
---

## Setup

```bash
git clone https://github.com/connect4deepak/LLM-Prompt-Firewall-with-RAG-based-Detection.git
cd LLM-Prompt-Firewall-with-RAG-based-Detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip3 install -r requirements.txt

#if getting error run below command
python -m pip install scikit-learn

```

---

## Execution — Step by Step

### Step 1 — Download and preprocess data

```bash
# With HuggingFace datasets (recommended, needs internet, ~10 min)
python3 scripts/01_download_data.py

# Offline / fast test using only built-in synthetic data
python3 scripts/01_download_data.py --offline
```

✅ Output: `data/processed/train.csv`, `val.csv`, `test.csv`

---

### Step 2 — Fine-tune DistilBERT

```bash
python3 scripts/02_train.py
```

- Downloads DistilBERT (~250 MB, one-time)
- Trains 3 epochs with early stopping
- **GPU:** ~5 min | **CPU:** ~60–90 min

✅ Output: `models/distilbert-firewall/`

---

### Step 3 — Build vector store

```bash
python3 scripts/03_build_vectorstore.py
```

- Embeds all adversarial prompts using sentence-transformers
- Persists ChromaDB index to `vectorstore/`
- Takes ~2–5 min

✅ Output: `vectorstore/`

---

### Step 4 — Full evaluation (generates all plots)

```bash
python3 scripts/04_evaluate.py
```

Produces in `results/`:

| File | Content |
|---|---|
| `00_class_distribution.png` | Dataset class balance |
| `01_model_comparison.png` | TF-IDF+LR vs DistilBERT vs DistilBERT+RAG |
| `02_confusion_matrix_classifier.png` | Classifier confusion matrix |
| `02_confusion_matrix_rag.png` | Full pipeline confusion matrix |
| `03_latency.png` | p50/p95 latency boxplot |
| `04_false_positives.png` | False positive analysis |
| `05_multilingual_bias.png` | Multilingual bias bar chart |
| `06_ablation.png` | Ablation study (k=1,3,5) |
| `07_token_importance_1.png` | Token importance for sample prompts |
| `ablation_results.csv` | Ablation numbers |
| `false_positives.csv` | FP prompt list |
| `multilingual_results.csv` | Multilingual results |

---

### Step 5 — Run the Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**

Three tabs:
- **🛡️ Firewall** — classify any prompt, see confidence bars and retrieved similar attacks
- **📊 Analytics** — live pie/bar/gauge charts of session activity
- **🔍 Explainability** — coloured token importance map + multilingual bias test

---

### Step 6 (optional) — FastAPI backend

```bash
uvicorn app.api:app --reload
# Swagger UI: http://localhost:8000/docs
```

```bash
# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ignore all previous instructions", "use_rag": true}'
```

---

### Step 7 (optional) — ngrok public URL for presentation

```bash
# Add NGROK_AUTH_TOKEN to .env first
streamlit run app/streamlit_app.py &
python -c "
from pyngrok import ngrok
from dotenv import load_dotenv; import os
load_dotenv()
ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN'))
print(ngrok.connect(8501))
input('Press Enter to stop')
"
```

---

## API Reference

```
GET  /health            → system status
POST /predict           → classify single prompt
POST /predict/batch     → classify list of prompts
```

---

## Expected Results

| Model | F1 (macro) | Precision | Recall |
|---|---|---|---|
| TF-IDF + LR (baseline) | ~0.72 | ~0.74 | ~0.71 |
| DistilBERT only | ~0.88 | ~0.89 | ~0.87 |
| DistilBERT + RAG (k=3) | ~0.91 | ~0.92 | ~0.90 |

*Results vary by dataset size and hardware.*

---

## Authors

- Student 1 — ML Engineer (classifier, evaluation)
- Student 2 — RAG Engineer (vector store, pipeline)
- Student 3 — Demo Lead (Streamlit, presentation)

Dublin Business School | MSc Artificial Intelligence | NLP Module
