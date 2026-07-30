"""
src/rag_pipeline.py
RAG layer using ChromaDB + sentence-transformers.

Stage 2 of the firewall:
  - Stores known adversarial prompts as embeddings in ChromaDB
  - When classifier confidence < CONFIDENCE_THRESHOLD,
    retrieve top-k similar attacks and make final decision
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (VECTORSTORE_DIR, EMBEDDING_MODEL, COLLECTION_NAME,
                    N_RESULTS, SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD,
                    LABEL_NAMES, LABEL_MAP)

class RAGPipeline:
    """
    ChromaDB-backed retrieval pipeline for adversarial prompt detection.

    Usage:
        rag = RAGPipeline()
        rag.build(adversarial_df)        # one-time setup
        result = rag.classify("some prompt")
    """

    def __init__(self):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ ChromaDB collection '{COLLECTION_NAME}' "
              f"({self.collection.count()} docs)")

    #  Build 
    def build(self, df: pd.DataFrame, batch_size: int = 256) -> None:
        """
        Index all adversarial prompts from df into ChromaDB.
        df must have columns: [text, label]
        Only JAILBREAK, PROMPT_INJECTION, PII_RISK are indexed (not SAFE).
        """
        adversarial = df[df["label"] != LABEL_MAP["SAFE"]].copy()
        adversarial = adversarial.drop_duplicates(subset=["text"]).reset_index(drop=True)

        print(f"\n🗄️  Building vector store with {len(adversarial)} adversarial prompts...")

        # Process in batches
        for start in range(0, len(adversarial), batch_size):
            batch = adversarial.iloc[start: start + batch_size]
            texts     = batch["text"].tolist()
            labels    = batch["label"].tolist()
            ids       = [f"doc_{start + i}" for i in range(len(texts))]
            metadatas = [{"label": LABEL_NAMES[l], "label_id": l} for l in labels]

            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"  Indexed {min(start + batch_size, len(adversarial))}"
                  f" / {len(adversarial)}", end="\r")

        print(f"\n  ✅ Vector store built: {self.collection.count()} documents")

    #  Query 
    def retrieve(self, prompt: str, n: int = N_RESULTS) -> List[Dict]:
        """
        Find the n most similar known adversarial prompts.
        Returns list of dicts with: text, label, distance, similarity
        """
        embedding = self.embedder.encode([prompt]).tolist()
        results   = self.collection.query(
            query_embeddings=embedding,
            n_results=min(n, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":       doc,
                "label":      meta["label"],
                "distance":   round(dist, 4),
                "similarity": round(1 - dist, 4),   # cosine: similarity = 1 - distance
            })
        return hits

    #  Full classification 
    def classify(self,
                 prompt: str,
                 classifier_result: Optional[Dict] = None) -> Dict:
        """
        Combine classifier result with RAG retrieval to make final verdict.

        Logic:
          1. If classifier confidence ≥ CONFIDENCE_THRESHOLD → trust classifier
          2. Else → retrieve similar attacks:
               - If top hit distance ≤ SIMILARITY_THRESHOLD → use retrieved label
               - Else → fall back to classifier
        """
        # Retrieve regardless (for explainability display)
        hits = self.retrieve(prompt)
        top_hit = hits[0] if hits else None

        if classifier_result is None:
            # RAG-only mode
            if top_hit and top_hit["distance"] <= SIMILARITY_THRESHOLD:
                verdict = top_hit["label"]
                method  = "RAG"
                blocked = (verdict != "SAFE")
            else:
                verdict = "SAFE"
                method  = "RAG_NO_MATCH"
                blocked = False
        else:
            conf = classifier_result["confidence"]

            if conf >= CONFIDENCE_THRESHOLD:
                # High-confidence classifier
                verdict = classifier_result["label"]
                method  = "CLASSIFIER"
                blocked = (verdict != "SAFE")
            elif top_hit and top_hit["distance"] <= SIMILARITY_THRESHOLD:
                # RAG overrides uncertain classifier
                verdict = top_hit["label"]
                method  = "RAG_OVERRIDE"
                blocked = (verdict != "SAFE")
            else:
                # Low confidence, no RAG match → trust classifier
                verdict = classifier_result["label"]
                method  = "CLASSIFIER_FALLBACK"
                blocked = (verdict != "SAFE")

        return {
            "verdict":     verdict,
            "blocked":     blocked,
            "method":      method,
            "similar_attacks": hits,
            "classifier":  classifier_result,
        }
