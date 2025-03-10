import pandas as pd

def calculate_revenue_at_risk(data):
    data['Revenue at Risk'] = data['Customer Count'] * data['Avg Monthly Revenue'] * data['Predicted Churn Risk']
    return data

def calculate_revenue_protected(data):
    data['Revenue Protected'] = data['Revenue at Risk'] * (1 - data['Predicted Churn Risk'])
    return data

def calculate_roi(data, investment_cost_per_location=100000):
    data['Investment Cost'] = investment_cost_per_location
    data['Revenue Saved'] = data['Revenue Protected']
    data['ROI'] = (data['Revenue Saved'] - data['Investment Cost']) / data['Investment Cost']
    return data