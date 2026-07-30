"""
scripts/03_build_vectorstore.py
Build ChromaDB vector store from adversarial training data
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from config import PROCESSED_DIR, LABEL_MAP

if __name__ == "__main__":
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    rag   = RAGPipeline()
    rag.build(train)
    print("\n✅ Step 3 complete. Run scripts/04_evaluate.py next.")
