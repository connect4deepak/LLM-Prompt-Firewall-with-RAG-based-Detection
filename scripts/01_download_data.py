"""
scripts/01_download_data.py
Download and merge all datasets, save to data/processed/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.data_loader  import load_all_datasets
from src.preprocessor import preprocess_pipeline
from config import RAW_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Use only synthetic data (no HuggingFace download)")
    args = parser.parse_args()

    df = load_all_datasets(use_hf=not args.offline)
    df.to_csv(RAW_DIR / "all_data.csv", index=False)
    print(f"✅ Raw data saved to {RAW_DIR / 'all_data.csv'}")

    train, val, test = preprocess_pipeline(df)
    print("\n✅ Step 1 complete. Run scripts/02_train.py next.")
