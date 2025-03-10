import streamlit as st
import pandas as pd
import numpy as np
import folium
import shap
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure Streamlit UI
st.set_page_config(layout="wide")
st.image("https://www.c3.ai/wp-content/uploads/2020/10/c3ai-logo.svg", width=200)
st.title("C3AI Telecom Churn Management System")
st.write("Leverage AI and ML insights to manage telecom churn risk effectively.")

# Define dataset ONCE
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 37.7749, 42.3601, 47.6062, 32.7767, 25.7617],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -122.4194, -71.0589, -122.3321, -96.7970, -80.1918],
    'Customer Count': [1000, 900, 850, 800, 750, 700, 650, 600, 550, 500],
    'Avg Monthly Revenue': [60, 55, 50, 65, 45, 52, 48, 55, 58, 62],
    'Network Performance Score': [8, 7, 6, 9, 5, 6, 8, 7, 5, 6],
    'Predicted Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Recommended Action': ['Monitor', 'Intervene', 'Urgent Action', 'Monitor', 'Critical Action',
                           'Intervene', 'Monitor', 'Urgent Action', 'Urgent Action', 'Critical Action'],
    'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6, 0.4, 0.2, -0.1, 0.6, -0.3]
})

# Add calculated columns
data['Revenue at Risk'] = data['Customer Count'] * data['Avg Monthly Revenue'] * data['Predicted Churn Risk']
data['Revenue Protected'] = data['Revenue at Risk'] * (1 - data['Predicted Churn Risk'])

# Sidebar filters
st.sidebar.header("Filters")
location_filter = st.sidebar.multiselect("Location", data['Location'].unique(), data['Location'].unique())
action_filter = st.sidebar.multiselect("Recommended Action", data['Recommended Action'].unique(), data['Recommended Action'].unique())
performance_filter = st.sidebar.slider("Network Performance Score", 1, 10, (5, 9))

# Apply filters
filtered_data = data[
    (data['Location'].isin(location_filter)) &
    (data['Recommended Action'].isin(action_filter)) &
    (data['Network Performance Score'].between(performance_filter[0], performance_filter[1]))
]

# Function for KPI Display
def display_kpis():
    st.subheader("📊 Key Performance Indicators")
    cols = st.columns(5)
    kpi_values = [
        f"${filtered_data['Avg Monthly Revenue'].mean():.2f}",
        f"${filtered_data['Revenue at Risk'].sum():,.2f}",
        f"${filtered_data['Revenue Protected'].sum():,.2f}",
        f"{filtered_data['Network Performance Score'].mean():.2f}",
        f"{filtered_data['Predicted Churn Risk'].mean() * 100:.1f}%"
    ]
    kpi_names = ['Avg Monthly Revenue', 'Revenue at Risk', 'Revenue Protected', 'Avg Network Performance', 'Avg Churn Probability']
    
    for col, name, value in zip(cols, kpi_names, kpi_values):
        col.metric(name, value)

display_kpis()


st.subheader("📊 Feature Importance in Churn Prediction")

# Explain the model's predictions
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot SHAP summary
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)
# 📉 Churn Risk vs. Investment Chart
st.subheader("📉 Scenario Analysis: Network Investment Impact")
investment_levels = np.linspace(0, 500000, 100)
fig = px.line()
for location in filtered_data['Location'].unique():
    churn_risk = filtered_data.loc[filtered_data['Location'] == location, 'Predicted Churn Risk'].values[0]
    customer_count = filtered_data.loc[filtered_data['Location'] == location, 'Customer Count'].values[0]
    forecasted_churn_reduction = churn_risk * (1 - (investment_levels / 1000000))
    churn_reduction = forecasted_churn_reduction * customer_count
    fig.add_scatter(x=investment_levels, y=churn_reduction, mode='lines', name=location)
st.plotly_chart(fig)

# 📈 Clustering Churn Risk
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Predicted Churn Risk', 'Network Performance Score', 'Revenue at Risk']])
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

st.subheader("📈 Customer Segmentation (K-Means Clustering)")
fig = px.scatter(data, x="Network Performance Score", y="Predicted Churn Risk", color=data['Cluster'].astype(str),
                 size="Revenue at Risk", hover_name="Location", title="Customer Clusters by Churn Risk")
st.plotly_chart(fig)

# 🌍 Churn Risk Map
st.subheader("🌍 Churn Risk Map")
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles="CartoDB dark_matter")
for _, row in filtered_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['Predicted Churn Risk'] * 50,
        color="red" if row['Predicted Churn Risk'] > 0.2 else "orange",
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Location']}: {row['Predicted Churn Risk']*100:.1f}% risk"
    ).add_to(m)
folium_static(m)

# 🚨 Anomaly Detection
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(data[['Predicted Churn Risk', 'Network Performance Score', 'Revenue at Risk']])
data['Anomaly'] = data['Anomaly'].apply(lambda x: '🔴 Anomaly' if x == -1 else '🟢 Normal')

st.subheader("🚨 Anomaly Detection in Churn Risk")
fig = px.scatter(data, x="Network Performance Score", y="Predicted Churn Risk", color="Anomaly",
                 size="Revenue at Risk", hover_name="Location", title="Anomaly Detection in Churn Risk")
st.plotly_chart(fig)

# 🧠 AI-Powered Churn Prediction
X = data[['Network Performance Score', 'Revenue at Risk', 'Sentiment Score']]
y = (data['Predicted Churn Risk'] > 0.3).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("🧠 AI-Powered Churn Prediction")
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")