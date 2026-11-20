import os
import json
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from src.preprocess import clean_text


def load_or_fetch_dataset(csv_path: str = "data/quora_questions.csv", sample_size: int = 12000) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Downloading Quora Question Pairs dataset from HuggingFace...")
        dataset = load_dataset("quora", split=f"train[:{sample_size}]")
        rows = []
        for item in dataset:
            q_texts = item["questions"]["text"]
            if len(q_texts) == 2:
                rows.append({
                    "question1": q_texts[0],
                    "question2": q_texts[1],
                    "is_duplicate": int(item["is_duplicate"]),
                })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} pairs to {csv_path}")

    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).drop_duplicates()
    return df


def train_and_evaluate():
    df = load_or_fetch_dataset()
    df["q1_clean"] = df["question1"].apply(clean_text)
    df["q2_clean"] = df["question2"].apply(clean_text)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_duplicate"])

    # 1. TF-IDF Baseline
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(pd.concat([train_df["q1_clean"], train_df["q2_clean"]]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    q1_tfidf = vectorizer.transform(test_df["q1_clean"])
    q2_tfidf = vectorizer.transform(test_df["q2_clean"])
    tfidf_sim = 1.0 - paired_cosine_distances(q1_tfidf, q2_tfidf)
    tfidf_preds = (tfidf_sim >= 0.62).astype(int)

    # 2. Sentence-BERT Semantic Model
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    q1_emb = sbert.encode(test_df["question1"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    q2_emb = sbert.encode(test_df["question2"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    sbert_sim = np.sum(q1_emb * q2_emb, axis=1)
    sbert_preds = (sbert_sim >= 0.75).astype(int)

    # 3. Compute & Save Metrics
    y_true = test_df["is_duplicate"].values
    metrics = {}
    for name, preds in [("TF-IDF + Cosine", tfidf_preds), ("Sentence-BERT", sbert_preds)]:
        metrics[name] = {
            "Accuracy": round(float(accuracy_score(y_true, preds)), 4),
            "Precision": round(float(precision_score(y_true, preds)), 4),
            "Recall": round(float(recall_score(y_true, preds)), 4),
            "F1-Score": round(float(f1_score(y_true, preds)), 4),
        }

    with open("models/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Metrics ===")
    print(pd.DataFrame(metrics).T)


if __name__ == "__main__":
    train_and_evaluate()
