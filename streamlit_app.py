import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score# Streamlit UI Configuration
st.set_page_config(layout="wide")
from sklearn.ensemble import IsolationForest
# Dashboard Title and C3.ai Logo
st.image("https://www.c3.ai/wp-content/uploads/2020/10/c3ai-logo.svg", width=200)
st.title("C3AI Telecom Churn Management")
st.write("Leverage AI and ML insights to manage telecom churn risk effectively.")


# Sample data preparation
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 37.7749, 42.3601, 47.6062, 32.7767, 25.7617],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -122.4194, -71.0589, -122.3321, -96.7970, -80.1918],
    'Customer Count': [1000, 900, 850, 800, 750, 700, 650, 600, 550, 500],
    'Avg Monthly Revenue': [60, 55, 50, 65, 45, 52, 48, 55, 58, 62],
    'Network Performance Score': [8, 7, 6, 9, 5, 6, 8, 7, 5, 6],
    'Predicted Churn Risk': [0.15, 0.20, 0.25, 0.10, 0.30, 0.22, 0.18, 0.20, 0.27, 0.25],
    'Recommended Action': ['High Priority', 'Medium Priority', 'Low Priority', 'High Priority', 'Medium Priority', 'Medium Priority', 'High Priority', 'Medium Priority', 'High Priority', 'Low Priority'],
    'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6, 0.4, 0.2, -0.1, 0.6, -0.3]
})


# Calculations for financial metrics
data['Revenue at Risk'] = data['Customer Count'] * data['Avg Monthly Revenue'] * data['Predicted Churn Risk']
data['Revenue Protected'] = data['Revenue at Risk'] * (1 - data['Predicted Churn Risk'])


# Sidebar filters
st.sidebar.header("Filters")
location_filter = st.sidebar.multiselect("Location", data['Location'].unique(), data['Location'].unique())
action_filter = st.sidebar.multiselect("Recommended Action", data['Recommended Action'].unique(), data['Recommended Action'].unique())
performance_filter = st.sidebar.slider("Network Performance Score", 1, 10, (5, 9))


filtered_data = data[
    (data['Location'].isin(location_filter)) &
    (data['Recommended Action'].isin(action_filter)) &
    (data['Network Performance Score'] >= performance_filter[0]) &
    (data['Network Performance Score'] <= performance_filter[1])
]


# KPIs at the top
st.subheader("📊 Key Performance Indicators")
cols = st.columns(5)
kpis = [
    f"${filtered_data['Avg Monthly Revenue'].mean():.2f}",
    f"${filtered_data['Revenue at Risk'].sum():,.2f}",
    f"${filtered_data['Revenue Protected'].sum():,.2f}",
    f"{filtered_data['Network Performance Score'].mean():.2f}",
    f"{filtered_data['Predicted Churn Risk'].mean() * 100:.1f}%"
]


for col, kpi_name, kpi in zip(cols, ['Avg Monthly Revenue', 'Revenue at Risk', 'Revenue Protected', 'Avg Network Performance', 'Avg Churn Probability'], kpis):
    col.metric(kpi_name, kpi)


# Scenario Analysis - Investment vs. Churn Risk Reduction
st.subheader("📉 Scenario Analysis: Network Investment Impact")
investment_levels = np.linspace(0, 500000, 100)
fig = px.line()
for location in filtered_data['Location'].unique():
    churn_risk = filtered_data.loc[filtered_data['Location'] == location, 'Predicted Churn Risk'].values[0]
    customer_count = filtered_data.loc[filtered_data['Location'] == location, 'Customer Count'].values[0]
    forecasted_churn_reduction = churn_risk * (1 - (investment_levels / 1000000))
    churn_reduction = forecasted_churn_reduction * customer_count
    fig.add_scatter(x=investment_levels, y=churn_reduction, mode='lines', name=location)
fig.update_layout(title='Forecasted Churn Probability by Network Investment',
                  xaxis_title='Investment Level ($)',
                  yaxis_title='Forecasted Churn Reduction')
st.plotly_chart(fig)


# Cluster Chart - Churn Risk vs Revenue
st.subheader("📈 Churn Risk Clusters")
fig = px.scatter(filtered_data, x="Network Performance Score", y="Predicted Churn Risk", size="Customer Count", color="Revenue at Risk",
                 hover_name="Location", title="Churn Risk Clusters")
st.plotly_chart(fig)


# Churn Risk Map
st.subheader("🌍 Churn Risk Map")
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles="CartoDB dark_matter")
for index, row in filtered_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['Predicted Churn Risk'] * 50,
        color="red" if row['Predicted Churn Risk'] > 0.2 else "orange",
        fill=True,
        fill_color="red" if row['Predicted Churn Risk'] > 0.2 else "orange",
        fill_opacity=0.6,
        popup=f"{row['Location']}: {row['Predicted Churn Risk']*100:.1f}% risk"
    ).add_to(m)
folium_static(m)




# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Predicted Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Recommended Action': ['Monitor', 'Intervene', 'Urgent Action', 'Monitor', 'Critical Action',
                           'Intervene', 'Monitor', 'Urgent Action', 'Urgent Action', 'Critical Action']
})
# Count alert severity levels


# Filter high-risk locations
high_risk_data = data[data['Predicted Churn Risk'] > 0.25]


# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Predicted Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Network Performance Score': [8, 7, 5, 9, 4, 6, 8, 5, 6, 3],
    'Recommended Action': ['Monitor', 'Intervene', 'Urgent Action', 'Monitor', 'Critical Action',
                           'Intervene', 'Monitor', 'Urgent Action', 'Urgent Action', 'Critical Action']
})


# Define alert levels based on churn risk
def get_alert_status(risk):
    if risk >= 0.4:
        return "🔴 Critical"
    elif risk >= 0.3:
        return "🟠 Urgent"
    elif risk >= 0.2:
        return "🟡 Intervene"
    else:
        return "🟢 Monitor"


data['Alert Status'] = data['Predicted Churn Risk'].apply(get_alert_status)


# Define alert details
def get_alert_details(row):
    if row['Alert Status'] == "🔴 Critical":
        return "Immediate action needed to prevent customer churn."
    elif row['Alert Status'] == "🟠 Urgent":
        return "High churn risk. Customer retention strategy required."
    elif row['Alert Status'] == "🟡 Intervene":
        return "Moderate churn risk. Consider network improvements."
    else:
        return "Low risk. Monitor for any upcoming changes."


data['Alert Details'] = data.apply(get_alert_details, axis=1)


# Streamlit UI
st.title("🚨 Proactive Action Center")
st.write("Identifying high churn risk locations and recommended actions.")


# Display alerts in table format
st.subheader("📊 Alert Status by Location")
st.write("Churn risk levels with recommended actions.")


# Apply conditional formatting
def color_alert(val):
    color = ""
    if "Critical" in val:
        color = "background-color: #ff4d4d; color: white;"  # Red for Critical
    elif "Urgent" in val:
        color = "background-color: #ff944d; color: white;"  # Orange for Urgent
    elif "Intervene" in val:
        color = "background-color: #ffd633; color: black;"  # Yellow for Intervene
    elif "Monitor" in val:
        color = "background-color: #70db70; color: black;"  # Green for Monitor
    return color


# Apply color formatting to Alert Status
styled_table = data.style.applymap(color_alert, subset=['Alert Status'])


# Display the table
st.dataframe(styled_table)




# Social Media Sentiment Analysis
st.subheader("🗣️ Social Media Sentiment")
st.bar_chart(filtered_data.set_index('Location')['Sentiment Score'])
# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Predicted Churn Risk': [0.15, 0.20, 0.25, 0.10, 0.30, 0.22, 0.18, 0.20, 0.27, 0.25],
    'Network Performance Score': [8, 7, 6, 9, 5, 6, 8, 7, 5, 6],
    'Revenue at Risk': [90000, 99000, 106250, 52000, 101250, 80000, 70200, 66000, 89000, 77500],
    'Recommended Action': ['High Priority', 'Medium Priority', 'Low Priority', 'High Priority',
                           'Medium Priority', 'Medium Priority', 'High Priority', 'Medium Priority',
                           'High Priority', 'Low Priority']
})


# Streamlit UI
st.title("📊 Churn Risk Data Table")
st.write("Overview of locations with churn risk analysis.")


# Display the table
st.dataframe(data)
# Data export
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name='filtered_telecom_churn_data.csv',
    mime='text/csv'
)


# Sample churn data
data = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'y': [0.15, 0.18, 0.21, 0.20, 0.25, 0.23, 0.30, 0.32, 0.28, 0.35, 0.33, 0.40]  # Churn probabilities over time
})




# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 6, 3],
    'Revenue': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 102000, 87000]
})


# Normalize data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Churn Risk', 'Network Performance', 'Revenue']])


# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)


# Streamlit UI
st.title("📊 Customer Segmentation (K-Means Clustering)")
st.write("Grouping locations into risk-based categories.")


# Cluster Visualization
fig = px.scatter(data, x="Network Performance", y="Churn Risk", color=data['Cluster'].astype(str),
                 size="Revenue", hover_name="Location", title="Customer Clusters by Churn Risk")
st.plotly_chart(fig)






# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.50, 0.10, 0.40, 0.22, 0.18, 0.30, 0.55, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 3, 6],
    'Revenue at Risk': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 135000, 87000]
})


# Train Isolation Forest model
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(data[['Churn Risk', 'Network Performance', 'Revenue at Risk']])


# Convert anomaly labels (-1 = anomaly, 1 = normal)
data['Anomaly'] = data['Anomaly'].apply(lambda x: '🔴 Anomaly' if x == -1 else '🟢 Normal')


# Streamlit UI
st.title("🚨 Anomaly Detection in Churn Risk")
st.write("Detecting unusual spikes in churn risk using Isolation Forest.")


# Display anomalies in a table
st.dataframe(data[['Location', 'Churn Risk', 'Network Performance', 'Revenue at Risk', 'Anomaly']])


# Anomaly visualization
fig = px.scatter(data, x="Network Performance", y="Churn Risk", color="Anomaly",
                 size="Revenue at Risk", hover_name="Location", title="Anomaly Detection in Churn Risk")
st.plotly_chart(fig)




# Sample data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 6, 3],
    'Revenue at Risk': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 102000, 87000],
    'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6, 0.4, 0.2, -0.1, 0.6, -0.3]
})


# Create binary churn labels (1 = high churn, 0 = low churn)
data['Churn Label'] = (data['Churn Risk'] > 0.3).astype(int)


# Train XGBoost model
X = data[['Network Performance', 'Revenue at Risk', 'Sentiment Score']]
y = data['Churn Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Streamlit UI
st.title("🧠 AI-Powered Churn Prediction (Sentiment + XGBoost)")
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")


# SHAP values interpretation
fig = px.scatter(data, x="Sentiment Score", y="Churn Risk", color="Churn Label",
                 size="Revenue at Risk", hover_name="Location",
                 title="Impact of Sentiment on Churn Risk")
st.plotly_chart(fig)






# Streamlit UI
st.title("📈 Churn Probability Forecasting")
st.write("Forecasting churn probabilities over time to anticipate risk.")


# Sample churn data
churn_data = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=12, freq='M'),
    'y': np.random.uniform(0.1, 0.5, 12)  # Simulating churn probabilities between 0.1 - 0.5
})


# Train Prophet model
model = Prophet()
model.fit(churn_data)


# Generate future dates for prediction
future = model.make_future_dataframe(periods=6, freq='M')  # Predict next 6 months
forecast = model.predict(future)


# Visualization
fig = px.line(forecast, x='ds', y='yhat', title="Churn Risk Forecast",
              labels={'ds': 'Date', 'yhat': 'Predicted Churn Probability'})


st.plotly_chart(fig)



