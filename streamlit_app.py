import streamlit as st
import pandas as pd
import numpy as np
import folium
import sklearn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_folium import folium_static
import logging

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Streamlit UI Setup & Custom CSS (adjust table container so horizontal scrolling is avoided)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="C3AI Telecom Churn Management", layout="wide")
st.markdown("""
   <style>
   .main { background-color: #f4f4f4; }
   /* Ensure tables use the full width */
   .css-1oe6wy2, .css-1d391kg { overflow-x: visible !important; }
   .st-table { width: 100% !important; }
   .stMarkdown table { width: 100% !important; }
   .stMarkdown th, .stMarkdown td { word-wrap: break-word; }
   </style>
""", unsafe_allow_html=True)
st.title("📊 C3AI Telecom Churn Management By Ali Albassam")
st.write("🔍 Leverage AI & ML insights to manage telecom churn risk effectively.")

# ------------------------------------------------------------------------------
# Global Data Loading & Common Filters
# ------------------------------------------------------------------------------
@st.cache_data
def load_dashboard_data():
    # Sample data for the main dashboard
    data = pd.DataFrame({
      'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                   'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
      'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 37.7749, 42.3601, 47.6062, 32.7767, 25.7617],
      'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -122.4194, -71.0589, -122.3321, -96.7970, -80.1918],
      'Customer Count': [1000, 900, 850, 800, 750, 700, 650, 600, 550, 500],
      'Avg Monthly Revenue': [60, 55, 50, 65, 45, 52, 48, 55, 58, 62],
      'Network Performance Score': [8, 7, 6, 9, 5, 6, 8, 7, 5, 6],
      'Predicted Churn Risk': [0.15, 0.20, 0.25, 0.10, 0.30, 0.22, 0.18, 0.20, 0.27, 0.25],
      'Recommended Action': ['High Priority', 'Medium Priority', 'Low Priority', 'High Priority', 
                             'Medium Priority', 'Medium Priority', 'High Priority', 'Medium Priority', 
                             'High Priority', 'Low Priority'],
      'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6, 0.4, 0.2, -0.1, 0.6, -0.3]
    })
    # Calculations for financial metrics
    data['Revenue at Risk'] = data['Customer Count'] * data['Avg Monthly Revenue'] * data['Predicted Churn Risk']
    data['Revenue Protected'] = data['Revenue at Risk'] * (1 - data['Predicted Churn Risk'])
    return data

def get_common_filters(data):
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
    return filtered_data

# Load global data and compute filtered data (filters available on every page)
common_data = load_dashboard_data()
filtered_data = get_common_filters(common_data)

# ------------------------------------------------------------------------------
# Table of Contents
# ------------------------------------------------------------------------------
st.markdown("## Table of Contents")
st.markdown("""
1. [Dashboard Overview](#dashboard-overview)
2. [Proactive Action Center](#proactive-action-center)
3. [Investment Optimization](#investment-optimization)
4. [Social Media Sentiment](#social-media-sentiment)
5. [Churn Risk Data Table](#churn-risk-data-table)
6. [Churn Probability Forecasting](#churn-probability-forecasting)
7. [Customer Segmentation](#customer-segmentation)
8. [Anomaly Detection](#anomaly-detection)
9. [AI-Powered Churn Prediction](#ai-powered-churn-prediction)
10. [Customer Lifetime Value (CLV) Prediction](#customer-lifetime-value-clv-prediction)
11. [Executive Workflow](#executive-workflow)
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Page Sections (Displayed Sequentially)
# ------------------------------------------------------------------------------

# 1. Dashboard Overview
st.markdown("## Dashboard Overview", unsafe_allow_html=True)
st.subheader("📊 Key Performance Indicators")
cols = st.columns(5)
kpis = [
    f"${filtered_data['Avg Monthly Revenue'].mean():.2f}",
    f"${filtered_data['Revenue at Risk'].sum():,.2f}",
    f"${filtered_data['Revenue Protected'].sum():,.2f}",
    f"{filtered_data['Network Performance Score'].mean():.2f}",
    f"{filtered_data['Predicted Churn Risk'].mean() * 100:.1f}%"
]
for col, kpi_name, kpi in zip(cols, 
                              ['Avg Monthly Revenue', 'Revenue at Risk', 'Revenue Protected', 
                               'Avg Network Performance', 'Avg Churn Probability'], kpis):
    col.metric(kpi_name, kpi)

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

st.subheader("📈 Churn Risk Clusters")
fig = px.scatter(filtered_data, x="Network Performance Score", y="Predicted Churn Risk", 
                 size="Customer Count", color="Revenue at Risk",
                 hover_name="Location", title="Churn Risk Clusters")
st.plotly_chart(fig)

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

# 2. Proactive Action Center
st.markdown("## Proactive Action Center", unsafe_allow_html=True)
st.write("Identifying high churn risk locations and recommended actions.")
data_alerts = common_data.copy()
def get_alert_status(risk):
    if risk >= 0.4:
        return "🔴 Critical"
    elif risk >= 0.3:
        return "🟠 Urgent"
    elif risk >= 0.2:
        return "🟡 Intervene"
    else:
        return "🟢 Monitor"
data_alerts['Alert Status'] = data_alerts['Predicted Churn Risk'].apply(get_alert_status)
def get_alert_details(row):
    if row['Alert Status'] == "🔴 Critical":
        return "Immediate action needed to prevent customer churn."
    elif row['Alert Status'] == "🟠 Urgent":
        return "High churn risk. Customer retention strategy required."
    elif row['Alert Status'] == "🟡 Intervene":
        return "Moderate churn risk. Consider network improvements."
    else:
        return "Low risk. Monitor for any upcoming changes."
data_alerts['Alert Details'] = data_alerts.apply(get_alert_details, axis=1)
st.subheader("📊 Alert Status by Location")
st.write("Churn risk levels with recommended actions.")
def color_alert(val):
    if "Critical" in val:
        return "background-color: #ff4d4d; color: white;"
    elif "Urgent" in val:
        return "background-color: #ff944d; color: white;"
    elif "Intervene" in val:
        return "background-color: #ffd633; color: black;"
    elif "Monitor" in val:
        return "background-color: #70db70; color: black;"
    return ""
styled_table = data_alerts.style.applymap(color_alert, subset=['Alert Status'])
# Limit vertical space by wrapping the table in a scrollable div
table_html = styled_table.to_html()
st.markdown(f"<div style='max-height:300px; overflow-y:auto'>{table_html}</div>", unsafe_allow_html=True)

# 3. Investment Optimization
st.markdown("## Investment Optimization", unsafe_allow_html=True)
investment_levels = np.linspace(0, 500000, 10)
rewards = [np.clip(0.05 * (inv / 500000), 0, 0.5) for inv in investment_levels]
if len(investment_levels) != len(rewards):
    st.error("Error: investment_levels and rewards must have the same length!")
fig = px.bar(x=investment_levels, y=rewards, labels={'x': 'Investment ($)', 'y': 'Churn Reduction'},
             title="RL-Optimized Investment for Churn Reduction")
st.plotly_chart(fig)

# 4. Social Media Sentiment
st.markdown("## Social Media Sentiment", unsafe_allow_html=True)
st.bar_chart(filtered_data.set_index('Location')['Sentiment Score'])

# 5. Churn Risk Data Table
st.markdown("## Churn Risk Data Table", unsafe_allow_html=True)
st.write("Overview of locations with churn risk analysis.")
st.table(filtered_data)
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name='filtered_telecom_churn_data.csv',
    mime='text/csv'
)

# 6. Churn Probability Forecasting
st.markdown("## Churn Probability Forecasting", unsafe_allow_html=True)
st.write("Forecasting churn probabilities over time to anticipate risk.")
churn_data = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=12, freq='M'),
    'y': np.random.uniform(0.1, 0.5, 12)
})
try:
    model = Prophet()
    model.fit(churn_data)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title="Churn Risk Forecast",
                  labels={'ds': 'Date', 'yhat': 'Predicted Churn Probability'})
    st.plotly_chart(fig)
except Exception as e:
    st.error("Error in forecasting: " + str(e))
    logging.error("Prophet forecasting error: %s", e)

# 7. Customer Segmentation
st.markdown("## Customer Segmentation", unsafe_allow_html=True)
st.write("Grouping locations into risk-based categories.")
data_seg = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 6, 3],
    'Revenue': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 102000, 87000]
})
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_seg[['Churn Risk', 'Network Performance', 'Revenue']])
kmeans = KMeans(n_clusters=3, random_state=42)
data_seg['Cluster'] = kmeans.fit_predict(scaled_data)
fig = px.scatter(data_seg, x="Network Performance", y="Churn Risk", 
                 color=data_seg['Cluster'].astype(str),
                 size="Revenue", hover_name="Location",
                 title="Customer Clusters by Churn Risk")
st.plotly_chart(fig)

# 8. Anomaly Detection
st.markdown("## Anomaly Detection", unsafe_allow_html=True)
st.write("Detecting unusual spikes in churn risk using Isolation Forest.")
data_anom = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.50, 0.10, 0.40, 0.22, 0.18, 0.30, 0.55, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 3, 6],
    'Revenue at Risk': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 135000, 87000]
})
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data_anom['Anomaly'] = iso_forest.fit_predict(data_anom[['Churn Risk', 'Network Performance', 'Revenue at Risk']])
data_anom['Anomaly'] = data_anom['Anomaly'].apply(lambda x: '🔴 Anomaly' if x == -1 else '🟢 Normal')
st.table(data_anom[['Location', 'Churn Risk', 'Network Performance', 'Revenue at Risk', 'Anomaly']])
fig = px.scatter(data_anom, x="Network Performance", y="Churn Risk", color="Anomaly",
                 size="Revenue at Risk", hover_name="Location", title="Anomaly Detection in Churn Risk")
st.plotly_chart(fig)

# 9. AI-Powered Churn Prediction
st.markdown("## AI-Powered Churn Prediction", unsafe_allow_html=True)
data_pred = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'San Francisco', 'Boston', 'Seattle', 'Dallas', 'Miami'],
    'Churn Risk': [0.15, 0.25, 0.35, 0.10, 0.40, 0.22, 0.18, 0.30, 0.27, 0.45],
    'Network Performance': [8, 7, 5, 9, 4, 6, 8, 5, 6, 3],
    'Revenue at Risk': [120000, 110000, 95000, 105000, 89000, 115000, 99000, 108000, 102000, 87000],
    'Sentiment Score': [0.8, 0.3, -0.2, 0.5, -0.6, 0.4, 0.2, -0.1, 0.6, -0.3]
})
data_pred['Churn Label'] = (data_pred['Churn Risk'] > 0.3).astype(int)
X = data_pred[['Network Performance', 'Revenue at Risk', 'Sentiment Score']]
y = data_pred['Churn Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")
fig = px.scatter(data_pred, x="Sentiment Score", y="Churn Risk", color="Churn Label",
                 size="Revenue at Risk", hover_name="Location",
                 title="Impact of Sentiment on Churn Risk")
st.plotly_chart(fig)

# 10. Customer Lifetime Value (CLV) Prediction
st.markdown("## Customer Lifetime Value (CLV) Prediction", unsafe_allow_html=True)
st.write("Predicting customer revenue potential.")
data_clv = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Churn Risk': np.random.uniform(0.1, 0.5, 5),
    'Network Performance': np.random.randint(3, 10, 5),
    'Revenue at Risk': np.random.randint(50000, 150000, 5)
})
data_clv['CLV'] = np.random.uniform(500, 5000, len(data_clv))
X = data_clv[['Churn Risk', 'Network Performance', 'Revenue at Risk']]
y = data_clv['CLV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor()
model.fit(X_train, y_train)
data_clv['Predicted CLV'] = model.predict(X)
fig = px.scatter(data_clv, x="Churn Risk", y="Predicted CLV", color="Network Performance",
                 title="Predicted CLV vs. Churn Risk")
st.plotly_chart(fig)

# 11. Executive Workflow
st.markdown("## Executive Workflow", unsafe_allow_html=True)
st.write("Leverage AI-driven insights to make instant strategic decisions.")
data_exec = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Churn Risk': np.random.uniform(0.3, 0.5, 5),
    'Network Performance': np.random.randint(3, 10, 5),
    'Revenue Impact': np.random.randint(50000, 150000, 5)
})
data_exec['AI_Urgency_Score'] = np.random.uniform(0.5, 1.0, len(data_exec))
data_exec['Recommended Action'] = [
    'Upgrade Network', 'Offer Discounts', 'Customer Outreach', 'Optimize Billing', 'Improve Support'
]
workflow_data = data_exec.copy()
workflow_data['Assigned Team'] = ['Network Ops', 'Marketing', 'Sales', 'Finance', 'Customer Support']
workflow_data['Status'] = ['Pending'] * len(data_exec)
st.subheader("🔍 AI-Powered Churn Risk Insights")
fig = px.scatter(workflow_data, x="Churn Risk", y="Revenue Impact", color="AI_Urgency_Score",
                 size="Revenue Impact", hover_name="Location",
                 title="AI Insights: Churn Risk vs Revenue Impact")
st.plotly_chart(fig)
st.subheader("🛠 AI-Driven Executive Decision Panel")
edited_data = st.data_editor(workflow_data, num_rows="dynamic", height=300)
st.subheader("⚡ Take Instant Action")
col1, col2, col3 = st.columns(3)
if col1.button("✅ Approve AI Recommendations"):
    edited_data['Status'] = "Approved"
    st.success("All AI-recommended actions have been approved!")
if col2.button("🚨 Escalate High-Risk Locations"):
    edited_data.loc[edited_data['Churn Risk'] > 0.4, 'Status'] = "Escalated"
    st.warning("High-risk locations have been escalated!")
if col3.button("🕒 Defer Low-Risk Actions"):
    edited_data.loc[edited_data['Churn Risk'] < 0.35, 'Status'] = "Deferred"
    st.info("Low-risk actions have been deferred!")
st.subheader("📊 Workflow Status Overview")
status_counts = edited_data['Status'].value_counts()
# Use a pie chart to display the workflow status distribution
fig_workflow = go.Figure(data=[go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.3)])
fig_workflow.update_layout(title="Workflow Status Distribution")
st.plotly_chart(fig_workflow)
st.download_button(
    label="📥 Download Workflow Data",
    data=edited_data.to_csv(index=False),
    file_name='executive_workflow.csv',
    mime='text/csv'
)



