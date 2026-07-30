"""
src/preprocessor.py
Clean, balance, and split the merged dataset.
"""

import re, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROCESSED_DIR, LABEL_NAMES, MAX_SAMPLES_PER_CLASS,
                    RANDOM_SEED, TEST_SIZE, VAL_SIZE)

def clean_text(text: str) -> str:
    """Normalise a single prompt string."""
    text = str(text).strip()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"http\S+", "[URL]", text)       # mask URLs
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip()

def balance_classes(df: pd.DataFrame,
                    max_per_class: int = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Undersample majority classes, oversample minority classes
    so every class has exactly min(actual, max_per_class) samples.
    """
    frames = []
    target = min(max_per_class,
                 df["label"].value_counts().max())   # don't upsample beyond largest

    for lbl in df["label"].unique():
        subset = df[df["label"] == lbl]
        if len(subset) >= target:
            frames.append(subset.sample(target, random_state=RANDOM_SEED))
        else:
            frames.append(resample(subset, replace=True,
                                   n_samples=target,
                                   random_state=RANDOM_SEED))
    return pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

def split_dataset(df: pd.DataFrame):
    """
    Stratified train / val / test split.
    Returns (train_df, val_df, test_df)
    """
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label"], random_state=RANDOM_SEED)

    relative_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=relative_val,
        stratify=train_val["label"], random_state=RANDOM_SEED)

    return train.reset_index(drop=True), \
           val.reset_index(drop=True), \
           test.reset_index(drop=True)

def preprocess_pipeline(df: pd.DataFrame):
    """Full preprocessing: clean → balance → split → save."""
    print("🔧 Preprocessing...")

    # 1. Clean
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 5].drop_duplicates(subset=["text"])
    print(f"  After cleaning: {len(df)} rows")

    # 2. Balance
    df = balance_classes(df)
    print(f"  After balancing: {len(df)} rows")

    # 3. Split
    train, val, test = split_dataset(df)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    # 4. Save
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR  / "val.csv",   index=False)
    test.to_csv(PROCESSED_DIR  / "test.csv",  index=False)
    print(f"  ✅ Saved splits to {PROCESSED_DIR}")

    return train, val, test

def load_splits():
    """Load pre-saved CSV splits."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test
