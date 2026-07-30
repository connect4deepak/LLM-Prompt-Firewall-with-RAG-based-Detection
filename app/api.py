"""
app/api.py
FastAPI backend — exposes the two-stage firewall as a REST API.

Endpoints:
  POST /predict       → classify a single prompt
  POST /predict/batch → classify a list of prompts
  GET  /health        → health check
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAVED_MODEL_DIR, VECTORSTORE_DIR
from src.classifier  import FirewallClassifier
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="LLM Prompt Firewall",
    description="Two-stage adversarial prompt detection: DistilBERT + RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
classifier: Optional[FirewallClassifier] = None
rag:        Optional[RAGPipeline]        = None

@app.on_event("startup")
def load_models():
    global classifier, rag
    try:
        classifier = FirewallClassifier(SAVED_MODEL_DIR)
        rag        = RAGPipeline()
        print("✅ Models loaded")
    except Exception as e:
        print(f"⚠️  Model load error: {e}")

# Schemas 
class PromptRequest(BaseModel):
    prompt: str
    use_rag: bool = True

class BatchRequest(BaseModel):
    prompts: List[str]
    use_rag: bool = True


# Endpoints 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "rag_loaded":        rag is not None,
        "vectorstore_docs":  rag.collection.count() if rag else 0,
    }

@app.post("/predict")
def predict(req: PromptRequest):
    if classifier is None:
        raise HTTPException(503, "Classifier not loaded. Run scripts/03_train.py first.")

    clf_result = classifier.predict(req.prompt)

    if req.use_rag and rag and rag.collection.count() > 0:
        result = rag.classify(req.prompt, clf_result)
    else:
        result = {
            "verdict":         clf_result["label"],
            "blocked":         clf_result["label"] != "SAFE",
            "method":          "CLASSIFIER",
            "similar_attacks": [],
            "classifier":      clf_result,
        }
    return result

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if classifier is None:
        raise HTTPException(503, "Classifier not loaded.")

    clf_results = classifier.predict(req.prompts)
    output = []
    for prompt, clf_result in zip(req.prompts, clf_results):
        if req.use_rag and rag and rag.collection.count() > 0:
            r = rag.classify(prompt, clf_result)
        else:
            r = {"verdict": clf_result["label"],
                 "blocked": clf_result["label"] != "SAFE",
                 "method":  "CLASSIFIER",
                 "similar_attacks": [],
                 "classifier": clf_result}
        output.append({"prompt": prompt, **r})
    return output