import pandas as pd
import os

# ==== Config ====
INPUT_PATH = 'data/transactions_with_anomalies.csv'
OUTPUT_PATH = 'data/transactions_with_explanations.csv'

def generate_explanation(row):
    if not row['is_anomaly']:
        return "Not anomalous."

    reasons = []

    # Rule 1: Unusual transaction amount
    if row['amount'] > 5000:
        reasons.append(f"high amount of £{row['amount']:.2f}")
    elif row['amount'] < 1:
        reasons.append("suspiciously low amount")

    # Rule 2: Late-night transaction
    if 'transaction_hour' in row and (row['transaction_hour'] < 6 or row['transaction_hour'] > 22):
        reasons.append(f"odd hour: {int(row['transaction_hour'])}:00")

    # Rule 3: High-risk merchant
    if row.get('high_risk_merchant') == 1:
        reasons.append("merchant flagged as high risk")

    # Rule 4: Distant from home
    if row.get('distance_from_home') == 1:
        reasons.append("customer was abroad")

    # Rule 5: Weekend pattern
    if row.get('weekend_transaction') == 1:
        reasons.append("occurred during weekend")

    if not reasons:
        return "Anomalous pattern detected with no obvious indicators."

    return "Anomalous due to: " + ", ".join(reasons) + "."

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ File not found: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"📦 Loaded {df.shape[0]} transactions.")

    # Generate explanations for all rows
    df['anomaly_explanation'] = df.apply(generate_explanation, axis=1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Explanations saved to: {OUTPUT_PATH}")
    print(df[['is_anomaly', 'amount', 'transaction_hour', 'anomaly_explanation']].head(5))

if __name__ == '__main__':
    main()
