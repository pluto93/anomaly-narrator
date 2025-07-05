# dashboard.py

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Anomaly Narrator", layout="wide")

# ----- LOAD DATA -----
@st.cache_data
def load_data():
    return pd.read_csv('data/transactions_with_clusters.csv')

df = load_data()

# Map fraud_cluster_label to human-readable themes
cluster_theme_map = {
    0: "Late-night + Abroad + High-Risk",
    1: "Late-night + Abroad",
    2: "Odd Hour Only",
    3: "Odd Hour + High-Risk",
    4: "Abroad Only"
}
df["cluster_theme"] = df["fraud_cluster_label"].map(cluster_theme_map)

# ----- SIDEBAR -----
st.sidebar.title("🔍 Filter Transactions")

show_anomalies = st.sidebar.checkbox("Show only anomalies", value=True)

amount_range = st.sidebar.slider("Amount Range (£)",
    float(df['amount'].min()), float(df['amount'].max()),
    (float(df['amount'].min()), float(df['amount'].max()))
)

hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

selected_categories = st.sidebar.multiselect(
    "Merchant Category",
    sorted(df['merchant_category'].dropna().unique()),
    default=None
)

selected_countries = st.sidebar.multiselect(
    "Country",
    sorted(df['country'].dropna().unique()),
    default=None
)

explanation_filter = st.sidebar.text_input("Keyword in Explanation (optional)")

# Cluster filters
cluster_options = sorted(df["fraud_cluster_label"].dropna().unique())
selected_cluster = st.sidebar.selectbox("Fraud Cluster Label", options=["All"] + list(cluster_options))

theme_options = sorted(df["cluster_theme"].dropna().unique())
selected_theme = st.sidebar.selectbox("Cluster Theme", ["All"] + theme_options)

# ----- FILTER DATA -----
filtered = df[
    (df['amount'] >= amount_range[0]) & (df['amount'] <= amount_range[1]) &
    (df['transaction_hour'] >= hour_range[0]) & (df['transaction_hour'] <= hour_range[1])
]

if show_anomalies:
    filtered = filtered[filtered['is_anomaly'] == True]

if selected_categories:
    filtered = filtered[filtered['merchant_category'].isin(selected_categories)]

if selected_countries:
    filtered = filtered[filtered['country'].isin(selected_countries)]

if explanation_filter:
    filtered = filtered[filtered['anomaly_explanation'].str.contains(explanation_filter, case=False, na=False)]

if selected_cluster != "All":
    filtered = filtered[filtered['fraud_cluster_label'] == selected_cluster]

if selected_theme != "All":
    filtered = filtered[filtered["cluster_theme"] == selected_theme]

# ----- MAIN HEADER -----
st.title("💡 Anomaly Narrator Dashboard")
st.markdown("Explore AI-detected anomalies with intuitive filters, natural language explanations, and clustered behavior patterns.")

# ----- METRICS -----
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Anomalies Detected", df['is_anomaly'].sum())
col3.metric("Filtered View", len(filtered))

# ----- SUMMARY -----
st.subheader("📌 Summary of Filtered Anomalies")
if not filtered.empty:
    st.markdown(f"""
    - **Avg Anomaly Score**: `{filtered['anomaly_score'].mean():.4f}`
    - **Max Transaction Amount**: `£{filtered['amount'].max():,.2f}`
    - **Top Merchant Category**: `{filtered['merchant_category'].mode()[0] if not filtered['merchant_category'].isna().all() else 'N/A'}`
    """)

# ----- CHARTS -----
st.subheader("📈 Anomaly Score Distribution")
score_chart = alt.Chart(filtered).mark_bar().encode(
    x=alt.X("anomaly_score", bin=alt.Bin(maxbins=40)),
    y='count()',
    tooltip=['count()']
).properties(height=200).interactive()
st.altair_chart(score_chart, use_container_width=True)

st.subheader("🕒 Anomalies by Hour")
hour_chart = alt.Chart(filtered).mark_bar().encode(
    x=alt.X("transaction_hour:O", title="Hour"),
    y='count()',
    tooltip=['count()']
).properties(height=200).interactive()
st.altair_chart(hour_chart, use_container_width=True)

st.subheader("🔢 Fraud Cluster Distribution")
cluster_counts = df["fraud_cluster_label"].value_counts().reset_index()
cluster_counts.columns = ["fraud_cluster_label", "count"]

cluster_chart = alt.Chart(cluster_counts).mark_bar().encode(
    x=alt.X("fraud_cluster_label:O", title="Cluster"),
    y=alt.Y("count:Q", title="Count"),
    tooltip=["count"]
).properties(height=250)
st.altair_chart(cluster_chart, use_container_width=True)

st.subheader("🌐 Cluster Theme Distribution")
theme_counts = df["cluster_theme"].value_counts().reset_index()
theme_counts.columns = ["Cluster Theme", "Count"]

theme_chart = alt.Chart(theme_counts).mark_bar().encode(
    x=alt.X("Cluster Theme", sort="-y"),
    y="Count",
    tooltip=["Count"]
).properties(height=250)
st.altair_chart(theme_chart, use_container_width=True)

# ----- EXAMPLES -----
st.subheader("🔍 Sample Explanations (Filtered)")
if not filtered.empty:
    st.dataframe(
        filtered[[
            "amount", "transaction_hour", "country", "merchant_category",
            "fraud_cluster_label", "cluster_theme", "anomaly_score", "anomaly_explanation"
        ]].sort_values('anomaly_score').head(10).reset_index(drop=True),
        use_container_width=True
    )

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_anomalies.csv',
        mime='text/csv',
    )
else:
    st.info("No results match the selected filters.")
