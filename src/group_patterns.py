# src/group_patterns.py

import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import joblib
import os

# ==== Config ====
INPUT_PATH = "data/transactions_with_explanations.csv"
SAVE_PATH = "data/transactions_with_clusters.csv"
MODEL_PATH = "models/kmeans_fraud_clusters.pkl"

def embed_explanations(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)

def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def main():
    print("📦 Loading data...")
    full_df = pd.read_csv(INPUT_PATH)
    anomalies_df = full_df[full_df["is_anomaly"] == True].copy()

    print(f"✅ Loaded {len(anomalies_df)} anomalous records for clustering.")

    print("🧠 Embedding explanations...")
    embeddings = embed_explanations(anomalies_df["anomaly_explanation"].fillna("No explanation").reset_index(drop=True))

    print("🔁 Clustering embeddings...")
    labels, kmeans_model = cluster_embeddings(embeddings)

    anomalies_df["fraud_cluster_label"] = labels

    # Merge cluster labels back to full dataframe on index
    full_df = full_df.merge(
        anomalies_df[["anomaly_explanation", "fraud_cluster_label"]],
        on="anomaly_explanation",
        how="left"
    )

    print(f"💾 Saving clustered full data to: {SAVE_PATH}")
    full_df.to_csv(SAVE_PATH, index=False)

    print(f"💾 Saving KMeans model to: {MODEL_PATH}")
    joblib.dump(kmeans_model, MODEL_PATH)

    print("✅ Clustering complete. Sample cluster breakdown:")
    print(anomalies_df["fraud_cluster_label"].value_counts())

if __name__ == "__main__":
    main()
