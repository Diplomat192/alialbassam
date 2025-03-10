import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def create_bubble_chart(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='Network Performance Score', y='Predicted Churn Risk', size='Customer Count',
                    hue='Recommended Action', sizes=(20, 200), alpha=0.6, palette='viridis')
    plt.title('Churn Risk Clusters by Network Performance')
    plt.xlabel('Network Performance Score')
    plt.ylabel('Predicted Churn Risk')
    plt.legend(title='Recommended Action')
    plt.show()

def create_heatmap(data):
    map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=5)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Revenue at Risk'] / 10000,
            popup=f"{row['Location']}: ${row['Revenue at Risk']:,.2f}",
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(map)
    return map

def create_scenario_analysis(data):
    investment_levels = np.linspace(0, 500000, 100)
    forecasted_churn_reduction = data['Predicted Churn Risk'] * (1 - (investment_levels / 1000000))

    plt.figure(figsize=(12, 8))
    for location in data['Location'].unique():
        churn_reduction = forecasted_churn_reduction * data.loc[data['Location'] == location, 'Customer Count'].values[0]
        plt.plot(investment_levels, churn_reduction, label=location)
    plt.title('Forecasted Churn Probability by Network Investment')
    plt.xlabel('Investment Level ($)')
    plt.ylabel('Forecasted Churn Reduction')
    plt.legend(title='City Names')
    plt.show()