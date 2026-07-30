"""
config.py — Central configuration for LLM Firewall project
All paths, model names, and hyperparameters live here.
"""

from pathlib import Path

#  Paths 
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = BASE_DIR / "models"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#  Labels 
LABEL_NAMES = {
    0: "SAFE",
    1: "JAILBREAK",
    2: "PROMPT_INJECTION",
    3: "PII_RISK",
}
LABEL_MAP   = {v: k for k, v in LABEL_NAMES.items()}
NUM_LABELS  = len(LABEL_NAMES)

LABEL_COLORS = {
    "SAFE":             "#22c55e",
    "JAILBREAK":        "#ef4444",
    "PROMPT_INJECTION": "#f97316",
    "PII_RISK":         "#a855f7",
}

# Classifier 
CLASSIFIER_MODEL = "distilbert-base-uncased"
SAVED_MODEL_DIR  = str(MODELS_DIR / "distilbert-firewall")
MAX_LENGTH       = 128
BATCH_SIZE       = 16
NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01

# RAG 
EMBEDDING_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME          = "adversarial_prompts"
N_RESULTS                = 3
SIMILARITY_THRESHOLD     = 0.45
CONFIDENCE_THRESHOLD     = 0.90

# Data 
MAX_SAMPLES_PER_CLASS = 1500
RANDOM_SEED           = 42
TEST_SIZE             = 0.15
VAL_SIZE              = 0.15
