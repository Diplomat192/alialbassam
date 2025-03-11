import streamlit as st
import pandas as pd
import numpy as np
import folium import sklearn 
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.image("c3-logo-70-127.png", width=200)
st.title("C3AI Telecom Churn Management")
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


# Sidebar Filters
st.sidebar.header("Filters")
location_filter = st.sidebar.multiselect("Location", data['Location'].unique(), data['Location'].unique())
action_filter = st.sidebar.multiselect("Recommended Action", data['Recommended Action'].unique(), data['Recommended Action'].unique())
performance_filter = st.sidebar.slider("Network Performance Score", 1, 10, (5, 9))


filtered_data = data[
    (data['Location'].isin(location_filter)) &
    (data['Recommended Action'].isin(action_filter)) &
    (data['Network Performance Score'].between(performance_filter[0], performance_filter[1]))
]


# Function to display KPIs
def display_kpis():
    st.subheader("📊 Key Performance Indicators")
    cols = st.columns(5)
    kpis = [
        f"${filtered_data['Avg Monthly Revenue'].mean():.2f}",
        f"${filtered_data['Revenue at Risk'].sum():,.2f}",
        f"${filtered_data['Revenue Protected'].sum():,.2f}",
        f"{filtered_data['Network Performance Score'].mean():.2f}",
        f"{filtered_data['Predicted Churn Risk'].mean() * 100:.1f}%"
    ]
    for col, name, value in zip(cols, ['Avg Monthly Revenue', 'Revenue at Risk', 'Revenue Protected', 'Avg Network Performance', 'Avg Churn Probability'], kpis):
        col.metric(name, value)


display_kpis()


# Scenario Analysis - Investment vs. Churn Risk Reduction
st.subheader("📉 Scenario Analysis: Network Investment Impact")
investment_levels = np.linspace(0, 500000, 100)
fig = px.line()
for location in filtered_data['Location'].unique():
    churn_risk = filtered_data.loc[filtered_data['Location'] == location, 'Predicted Churn Risk'].values[0]
    customer_count = filtered_data.loc[filtered_data['Location'] == location, 'Customer Count'].values[0]
    churn_reduction = churn_risk * (1 - (investment_levels / 1000000)) * customer_count
    fig.add_scatter(x=investment_levels, y=churn_reduction, mode='lines', name=location)
st.plotly_chart(fig)


# Churn Risk Clusters
st.subheader("📈 Churn Risk Clusters")
fig = px.scatter(filtered_data, x="Network Performance Score", y="Predicted Churn Risk", size="Customer Count", color="Revenue at Risk",
                 hover_name="Location", title="Churn Risk Clusters")
st.plotly_chart(fig)


# Churn Risk Map
st.subheader("🌍 Churn Risk Map")
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles="CartoDB dark_matter")
for _, row in filtered_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['Predicted Churn Risk'] * 50,
        color="red" if row['Predicted Churn Risk'] > 0.2 else "orange",
        fill=True,
        popup=f"{row['Location']}: {row['Predicted Churn Risk']*100:.1f}% risk"
    ).add_to(m)
folium_static(m)


# AI-Powered Churn Prediction
st.subheader("🧠 AI-Powered Churn Prediction")
X = data[['Network Performance Score', 'Revenue at Risk', 'Sentiment Score']]
y = (data['Predicted Churn Risk'] > 0.3).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")


# Forecasting Churn Probability
st.subheader("📈 Churn Probability Forecasting")
churn_data = pd.DataFrame({'ds': pd.date_range(start='2024-01-01', periods=12, freq='M'),
                           'y': np.random.uniform(0.1, 0.5, 12)})
prophet_model = Prophet()
prophet_model.fit(churn_data)
future = prophet_model.make_future_dataframe(periods=6, freq='M')
forecast = prophet_model.predict(future)
fig = px.line(forecast, x='ds', y='yhat', title="Churn Risk Forecast",
              labels={'ds': 'Date', 'yhat': 'Predicted Churn Probability'})
st.plotly_chart(fig)

