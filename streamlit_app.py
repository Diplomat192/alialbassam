import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🎯 **Dashboard Title**
st.title("C3AI Telecom Churn Management")
st.write("Leverage AI and ML insights to manage telecom churn risk effectively.")

# 📌 **Sample Dataset**
churn_data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
    'Customer Count': [1000, 900, 850, 800, 750],
    'Avg Monthly Revenue': [60, 55, 50, 65, 45],
    'Network Performance Score': [8, 7, 6, 9, 5],
    'Predicted Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40],
    'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6]
})

# 📌 **Financial Metrics**
churn_data['Revenue at Risk'] = churn_data['Customer Count'] * churn_data['Avg Monthly Revenue'] * churn_data['Predicted Churn Risk']
churn_data['Revenue Protected'] = churn_data['Revenue at Risk'] * (1 - churn_data['Predicted Churn Risk'])

# 📌 **AI-Powered Executive Workflow Action Center**
st.title("📊 AI-Powered Executive Workflow Action Center")
st.write("Leverage AI-driven insights to make instant strategic decisions.")

# 📌 **AI-Powered Risk Scoring Model (XGBoost)**
churn_data['AI_Urgency_Score'] = np.random.uniform(0.5, 1.0, len(churn_data))  # Simulated urgency score (1 = high)

# 📌 **AI-Generated Recommended Actions**
churn_data['Recommended Action'] = [
    'Upgrade Network', 'Offer Discounts', 'Customer Outreach', 'Optimize Billing', 'Improve Support'
]

# **Executive Decision Workflow**
workflow_data = churn_data.copy()
workflow_data['Assigned Team'] = ['Network Ops', 'Marketing', 'Sales', 'Finance', 'Customer Support']
workflow_data['Status'] = ['Pending'] * len(churn_data)

# 📌 **Display AI Recommendations & Urgency Levels**
st.subheader("🔍 AI-Powered Churn Risk Insights")
fig = px.scatter(workflow_data, x="Predicted Churn Risk", y="Revenue at Risk", color="AI_Urgency_Score",
                 size="Revenue at Risk", hover_name="Location",
                 title="AI Insights: Churn Risk vs Revenue Impact")
st.plotly_chart(fig)

# 📌 **Executive Decision Panel**
st.subheader("🛠 AI-Driven Executive Decision Panel")
edited_data = st.data_editor(workflow_data, num_rows="dynamic", height=300)

# 📌 **One-Click Executive Actions**
st.subheader("⚡ Take Instant Action")
col1, col2, col3 = st.columns(3)

if col1.button("✅ Approve AI Recommendations"):
    edited_data['Status'] = "Approved"
    st.success("All AI-recommended actions have been approved!")

if col2.button("🚨 Escalate High-Risk Locations"):
    edited_data.loc[edited_data['Predicted Churn Risk'] > 0.4, 'Status'] = "Escalated"
    st.warning("High-risk locations have been escalated!")

if col3.button("🕒 Defer Low-Risk Actions"):
    edited_data.loc[edited_data['Predicted Churn Risk'] < 0.35, 'Status'] = "Deferred"
    st.info("Low-risk actions have been deferred.")

# 📌 **Task Status Summary**
st.subheader("📊 Workflow Status Overview")
status_counts = edited_data['Status'].value_counts()
st.bar_chart(status_counts)

# 📥 **Download Updated Workflow**
st.download_button(
    label="📥 Download Workflow Data",
    data=edited_data.to_csv(index=False),
    file_name='executive_workflow.csv',
    mime='text/csv'
)

# 🔎 **Anomaly Detection using Isolation Forest**
st.subheader("🚨 Anomaly Detection in Churn Risk")
iso_forest = IsolationForest(contamination=0.2, random_state=42)
workflow_data['Anomaly'] = iso_forest.fit_predict(workflow_data[['Predicted Churn Risk', 'Network Performance Score', 'Revenue at Risk']])
workflow_data['Anomaly'] = workflow_data['Anomaly'].apply(lambda x: '🔴 Anomaly' if x == -1 else '🟢 Normal')

fig = px.scatter(workflow_data, x="Network Performance Score", y="Predicted Churn Risk",
                 color="Anomaly", size="Revenue at Risk", hover_name="Location",
                 title="Anomaly Detection in Churn Risk")
st.plotly_chart(fig)