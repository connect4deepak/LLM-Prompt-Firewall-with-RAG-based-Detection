"""
src/classifier.py
Fine-tunes DistilBERT for 4-class prompt classification.
Also handles inference (predict a single prompt or batch).
"""

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import (classification_report, f1_score,
                              precision_recall_fscore_support, accuracy_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (CLASSIFIER_MODEL, SAVED_MODEL_DIR, MAX_LENGTH,
                    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    NUM_LABELS, LABEL_NAMES, RANDOM_SEED)


#  PyTorch Dataset
class PromptDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | \
               {"labels": self.labels[idx]}

# Metrics for HuggingFace Trainer 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1,
            "precision": precision, "recall": recall}

# Training 
def train_classifier(train_df: pd.DataFrame,
                     val_df:   pd.DataFrame,
                     output_dir: str = SAVED_MODEL_DIR) -> None:
    """
    Fine-tune DistilBERT on the training split.
    Saves the best checkpoint (by F1) to output_dir.
    """
    print(f"\n🤖 Fine-tuning {CLASSIFIER_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL, num_labels=NUM_LABELS)

    train_dataset = PromptDataset(
        train_df["text"].tolist(), train_df["label"].tolist(),
        tokenizer, MAX_LENGTH)
    val_dataset = PromptDataset(
        val_df["text"].tolist(), val_df["label"].tolist(),
        tokenizer, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        seed=RANDOM_SEED,
        report_to="none",        # disable W&B for simplicity
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✅ Model saved to {output_dir}")


# Inference 
class FirewallClassifier:
    """Wraps the fine-tuned model for inference."""

    def __init__(self, model_dir: str = SAVED_MODEL_DIR):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(self.device)
        self.model.eval()
        print(f"✅ Classifier loaded from {model_dir} ({self.device})")

    def predict(self, text: Union[str, List[str]]):
        """
        Returns:
          label_name  (str)
          label_id    (int)
          confidence  (float 0-1)
          all_probs   (dict label_name → prob)
        """
        single = isinstance(text, str)
        texts  = [text] if single else text

        enc = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for p in probs:
            label_id   = int(np.argmax(p))
            confidence = float(p[label_id])
            all_probs  = {LABEL_NAMES[i]: float(p[i]) for i in range(NUM_LABELS)}
            results.append({
                "label":      LABEL_NAMES[label_id],
                "label_id":   label_id,
                "confidence": confidence,
                "all_probs":  all_probs,
            })

        return results[0] if single else results

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Run classifier on test set and return metrics + report."""
        preds   = self.predict(test_df["text"].tolist())
        y_pred  = [p["label_id"] for p in preds]
        y_true  = test_df["label"].tolist()
        report  = classification_report(
            y_true, y_pred,
            target_names=[LABEL_NAMES[i] for i in range(NUM_LABELS)],
            output_dict=True)
        return {
            "classification_report": report,
            "y_true": y_true,
            "y_pred": y_pred,
        }
