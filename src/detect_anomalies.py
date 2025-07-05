# detect_anomalies.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==== Config ====
DATA_PATH = 'data/synthetic_fraud_data.csv'  # Change if needed
MODEL_PATH = 'models/isolation_forest.pkl'
SAVE_PATH = 'data/transactions_with_anomalies.csv'
RANDOM_STATE = 42


def load_data(path):
    print("⚠️ Using only the first 20,000 rows to avoid memory issues.")
    df = pd.read_csv(path, nrows=50000)
    print(f"Loaded data: {df.shape}")
    return df


def preprocess(df):
    # Drop columns that won't help or crash the model
    drop_cols = [
        'transaction_id', 'customer_id', 'card_number', 'timestamp',
        'merchant', 'currency', 'ip_address', 'device_fingerprint',
        'velocity_last_hour'  # 👈 Drop the problematic column
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode booleans
    for col in ['card_present', 'weekend_transaction', 'distance_from_home', 'high_risk_merchant']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=[
        'merchant_category', 'merchant_type', 'country', 'city',
        'city_size', 'card_type', 'device', 'channel'
    ], drop_first=True)

    # Separate fraud labels if present
    labels = df['is_fraud'] if 'is_fraud' in df.columns else None
    df = df.drop(columns=['is_fraud'], errors='ignore')

    return df, labels


def detect_anomalies(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=RANDOM_STATE)
    model.fit(X_scaled)

    preds = model.predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)  # Lower = more anomalous

    return preds, scores, model


def main():
    # Step 1: Load data
    df_raw = load_data(DATA_PATH, n_rows=50000)

    # Step 2: Preprocess
    df_processed, true_labels = preprocess(df_raw.copy())

    # Step 3: Detect anomalies
    preds, scores, model = detect_anomalies(df_processed)

    # Step 4: Save results
    df_raw['anomaly_score'] = scores
    df_raw['is_anomaly'] = (preds == -1)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    df_raw.to_csv(SAVE_PATH, index=False)

    print(f"✅ Anomalies detected and saved to: {SAVE_PATH}")
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(df_raw[['is_anomaly', 'anomaly_score']].value_counts())


if __name__ == '__main__':
    main()
