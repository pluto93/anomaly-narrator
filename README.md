# Anomaly Narrator

> AI-powered anomaly detection meets explainable insights — uncover and understand suspicious financial transactions.

View the dashboard here: https://anomaly-narrator.streamlit.app/

<img width="1430" alt="Screenshot 2025-07-05 at 10 49 55 pm" src="https://github.com/user-attachments/assets/e8ede515-e8ca-4650-8d19-cc1405d2f2ca" />

---

## 🔍 Project Overview

**Anomaly Narrator** is a full-stack data analytics + AI project that:
- Detects anomalous financial transactions using unsupervised learning
- Auto-generates **natural language explanations** for anomalies
- Presents everything in an interactive **Streamlit dashboard** with filters and visuals

Built to showcase **explainable AI (XAI)** + anomaly detection in a business-relevant context like fraud detection or transaction monitoring.

---
## 🚀 Features

- **Anomaly Detection**: Flags unusual transactions using statistical or model-based methods  
- **Natural Language Explanations**: Rule-based logic turns anomalies into readable reasons  
- **Semantic Clustering**: Similar explanations are grouped using Sentence-BERT + KMeans  
- **Interactive Dashboard**: Filter by time, country, cluster, explanation, and more in Streamlit  

---

## 🛠️ How It Works

### 🧪 1. Exploratory Data Analysis (`01_EDA.ipynb`)
- Understand key features of 20,000 synthetic financial transactions
- Explore fraud patterns, transaction time trends, merchant categories, etc.

### 🤖 2. Anomaly Detection (`src/detect_anomalies.py`)
- Uses `IsolationForest` to detect statistical outliers based on features like:
  - Transaction amount, time, location, card type, device, and more
- Outputs:
  - `is_anomaly`: Boolean flag
  - `anomaly_score`: Model's confidence

### 🧠 3. Explanation Generation (`src/explain_anomalies.py`)
- Rule-based logic generates simple, readable reasons like:
  > "Anomalous due to: odd hour: 0:00, merchant flagged as high-risk."

### 📊 4. Streamlit Dashboard (`app/dashboard.py`)
- Interactive UI to explore anomalies:
  - Filter by amount, country, merchant type, hour
  - Search keywords in explanations
  - Visuals: anomaly score histogram, hourly pattern chart
  - CSV download of filtered results

---
## ⚙️ Setup Instructions

### 1. 🔧 Install dependencies

pip install -r requirements.txt

### 2. 🛠️ Run preprocessing pipeline

python src/detect_anomalies.py
python src/explain_anomalies.py
python src/group_patterns.py

This will generate the processed dataset transactions_with_clusters.csv.

### 3. ▶️ Launch the dashboard

streamlit run app/dashboard.py

## 🧰 Tools & Libraries Used

| Tool                            | Purpose                           |
|----------------------------------|-----------------------------------|
| `pandas`, `numpy`               | Data manipulation                 |
| `scikit-learn`                  | Isolation Forest model            |
| `Altair`, `matplotlib`, `seaborn` | Visualization                     |
| `Streamlit`                     | Interactive dashboard             |
| `joblib`                        | Model saving                      |
| `sentence-transformers`         | Pattern clustering                |

## 📌 Example Use Cases

Banking: Fraud team reviews and prioritizes alerts with explanations

Insurance: Detects anomalous claim patterns

Ecommerce: Identifies suspicious orders or chargebacks

## 📸 Screenshots

<img width="1430" alt="Screenshot 2025-07-05 at 10 49 55 pm" src="https://github.com/user-attachments/assets/e7564e2f-2ad3-4d53-a652-3ab63d4edffd" />

<img width="1426" alt="Screenshot 2025-07-05 at 10 50 07 pm" src="https://github.com/user-attachments/assets/10f35998-10d1-4bd6-ba07-1f36322e6a2f" />

<img width="1431" alt="Screenshot 2025-07-05 at 10 50 15 pm" src="https://github.com/user-attachments/assets/df408bac-114e-4d30-ac2a-20668c4ca16f" />
