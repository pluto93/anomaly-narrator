# inspect_clusters.py

import pandas as pd

df = pd.read_csv("data/transactions_with_clusters.csv")

# Sample explanations per cluster
for cluster in sorted(df["fraud_cluster_label"].dropna().unique()):
    print(f"\n--- Cluster {cluster} ---")
    sample = df[df["fraud_cluster_label"] == cluster]["anomaly_explanation"].dropna().sample(10)
    for i, explanation in enumerate(sample, 1):
        print(f"{i}. {explanation}")
