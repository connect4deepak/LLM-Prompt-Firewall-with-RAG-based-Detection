"""
scripts/02_train.py
Fine-tune DistilBERT classifier
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import load_splits
from src.classifier   import train_classifier

if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    train_classifier(train, val)
    print("\n✅ Step 2 complete. Run scripts/03_build_vectorstore.py next.")
