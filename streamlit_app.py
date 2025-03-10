import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
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

# Calculate Revenue at Risk
data['Revenue at Risk'] = data['Customer Count'] * data['Avg Monthly Revenue'] * data['Predicted Churn Risk']

# Display Data
st.subheader("📊 Processed Telecom Churn Data")
st.dataframe(data)

# Churn Risk Scatter Plot
st.subheader("📈 Churn Risk vs Network Performance")
fig, ax = plt.subplots()
sns.scatterplot(data=data, x="Network Performance Score", y="Predicted Churn Risk", size="Customer Count", hue="Revenue at Risk", palette="coolwarm", alpha=0.7, ax=ax)
plt.xlabel("Network Performance Score")
plt.ylabel("Predicted Churn Risk")
plt.title("Churn Risk Clusters")
st.pyplot(fig)

# Map Visualization
st.subheader("🌍 Churn Risk by Location")
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
for index, row in data.iterrows():
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
