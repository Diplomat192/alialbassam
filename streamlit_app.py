import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from textblob import TextBlob

st.title("📊 Telecom Churn Analysis Dashboard")
st.write(
    "This dashboard uses ML and AI to help executives make data-driven decisions."
)

# Sample Data
data = pd.DataFrame({
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
    'Customer Count': [1000, 1200, 1100, 900, 850, 950, 800, 750, 700, 650],
    'Avg Monthly Revenue': [50, 45, 55, 60, 40, 50, 30, 35, 45, 50],
    'Network Performance Score': [8, 7, 9, 6, 5, 7, 6, 8, 7, 7],
    'Predicted Churn Risk': [0.1, 0.15, 0.2, 0.25, 0.1, 0.05, 0.3, 0.2, 0.15, 0.1],
    'Recommended Action': ['High Priority', 'Medium Priority', 'Low Priority', 'High Priority', 'Medium Priority',
                           'Medium Priority', 'High Priority', 'Low Priority', 'Medium Priority', 'Low Priority']
})

# Data Processing
# Remove duplicates and aggregate data by city
data_agg = data.groupby('Location').agg({
    'Customer Count': 'sum',
    'Avg Monthly Revenue': lambda x: np.average(x, weights=data.loc[x.index, 'Customer Count']),
    'Network Performance Score': lambda x: np.average(x, weights=data.loc[x.index, 'Customer Count']),
    'Predicted Churn Risk': lambda x: np.average(x, weights=data.loc[x.index, 'Customer Count']),
    'Latitude': 'first',
    'Longitude': 'first',
    'Recommended Action': 'first'
}).reset_index()

# Calculated Metrics
data_agg['Revenue at Risk'] = data_agg['Customer Count'] * data_agg['Avg Monthly Revenue'] * data_agg['Predicted Churn Risk']
data_agg['Revenue Protected'] = data_agg['Revenue at Risk'] * (1 - data_agg['Predicted Churn Risk'])

# ROI Calculation
investment_cost_per_location = 100000
data_agg['Investment Cost'] = investment_cost_per_location
data_agg['Revenue Saved'] = data_agg['Revenue Protected']
data_agg['ROI'] = (data_agg['Revenue Saved'] - data_agg['Investment Cost']) / data_agg['Investment Cost']

# Dummy Social Media Sentiment Analysis
def generate_dummy_sentiment():
    return np.random.uniform(-1, 1)

# Get sentiment scores for each location
for index, row in data_agg.iterrows():
    avg_sentiment = generate_dummy_sentiment()
    data_agg.at[index, 'Sentiment Score'] = avg_sentiment

# Proactive Alerts
def send_alert(location, issue):
    st.warning(f"Proactive Alert: {issue} in {location}")

# Check for network performance deterioration and increased churn risks
for index, row in data_agg.iterrows():
    if row['Network Performance Score'] < 5:
        send_alert(row['Location'], "Network Performance Score below 5")
    if row['Predicted Churn Risk'] > 0.8:
        send_alert(row['Location'], "Predicted Churn Risk above 80%")

# KPI Callout Cards
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue at Risk", f"${data_agg['Revenue at Risk'].sum():,.2f}")
col2.metric("Total Revenue Protected", f"${data_agg['Revenue Protected'].sum():,.2f}")
col3.metric("Average Network Performance Score", f"{data_agg['Network Performance Score'].mean():.2f}")
col4.metric("Average ROI", f"{data_agg['ROI'].mean() * 100:.2f}%")

# Display Data
st.subheader("📊 Processed Telecom Churn Data")
st.dataframe(data_agg)

# Churn Risk Scatter Plot
st.subheader("📈 Churn Risk vs Network Performance")
fig = px.scatter(data_agg, x="Network Performance Score", y="Predicted Churn Risk", size="Customer Count", color="Revenue at Risk",
                 hover_name="Location", title="Churn Risk Clusters")
st.plotly_chart(fig)

# Map Visualization
st.subheader("🌍 Churn Risk by Location")
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
for index, row in data_agg.iterrows():
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

# Scenario Analysis for Investment Impact
st.subheader("📉 Scenario Analysis for Investment Impact")
investment_levels = np.linspace(0, 500000, 100)

fig = px.line()
for location in data_agg['Location'].unique():
    churn_risk = data_agg.loc[data_agg['Location'] == location, 'Predicted Churn Risk'].values[0]
    customer_count = data_agg.loc[data_agg['Location'] == location, 'Customer Count'].values[0]
    forecasted_churn_reduction = churn_risk * (1 - (investment_levels / 1000000))
    churn_reduction = forecasted_churn_reduction * customer_count
    fig.add_scatter(x=investment_levels, y=churn_reduction, mode='lines', name=location)
fig.update_layout(title='Forecasted Churn Probability by Network Investment',
                  xaxis_title='Investment Level ($)',
                  yaxis_title='Forecasted Churn Reduction')
st.plotly_chart(fig)

# Filters
st.sidebar.header("Filters")
selected_action = st.sidebar.selectbox("Select Recommended Action", data_agg['Recommended Action'].unique())
filtered_data = data_agg[data_agg['Recommended Action'] == selected_action]

st.subheader(f"Filtered Data: {selected_action}")
st.dataframe(filtered_data)

# Save the processed data for further analysis if needed
data_agg.to_csv('processed_telecom_churn_data.csv', index=False)

# Provide a download link for the CSV file
st.download_button(
    label="Download Report as CSV",
    data=data_agg.to_csv(index=False),
    file_name='processed_telecom_churn_data.csv',
    mime='text/csv'
)