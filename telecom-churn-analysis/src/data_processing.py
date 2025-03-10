import pandas as pd
import numpy as np

def process_data(data):
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
    
    return data_agg

def calculate_revenue_metrics(data_agg):
    # Calculated Metrics
    data_agg['Revenue at Risk'] = data_agg['Customer Count'] * data_agg['Avg Monthly Revenue'] * data_agg['Predicted Churn Risk']
    data_agg['Revenue Protected'] = data_agg['Revenue at Risk'] * (1 - data_agg['Predicted Churn Risk'])
    
    return data_agg