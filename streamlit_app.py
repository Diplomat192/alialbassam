import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from textblob import TextBlob
import tweepy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Social Media Sentiment Analysis
# Set up Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Set up tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def get_tweets(query, count=100):
    tweets = []
    try:
        fetched_tweets = api.search(q=query, count=count)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet['text'] = tweet.text
            parsed_tweet['sentiment'] = TextBlob(tweet.text).sentiment.polarity
            tweets.append(parsed_tweet)
    except tweepy.TweepError as e:
        print(f"Error : {str(e)}")
    return tweets

# Get sentiment scores for each location
for index, row in data_agg.iterrows():
    tweets = get_tweets(query=row['Location'] + ' cellular reception', count=100)
    if tweets:
        avg_sentiment = np.mean([tweet['sentiment'] for tweet in tweets])
        data_agg.at[index, 'Sentiment Score'] = avg_sentiment
    else:
        data_agg.at[index, 'Sentiment Score'] = 0

# Proactive Alerts
def send_alert(location, issue):
    sender_email = "you@example.com"
    receiver_email = "stakeholder@example.com"
    password = "your_password"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Proactive Alert: Network Issue in " + location
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"""
    Hi Team,

    Please be informed that there is a network issue in {location}. The following issue has been detected:
    {issue}

    Kindly take the necessary actions.

    Best Regards,
    Network Monitoring Team
    """
    part1 = MIMEText(text, "plain")
    message.attach(part1)

    with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Check for network performance deterioration and increased churn risks
for index, row in data_agg.iterrows():
    if row['Network Performance Score'] < 6:
        send_alert(row['Location'], "Network Performance Score below 6")
    if row['Predicted Churn Risk'] > 0.2:
        send_alert(row['Location'], "Predicted Churn Risk above 20%")

# Helper function to create a bubble chart
def create_bubble_chart(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='Network Performance Score', y='Predicted Churn Risk', size='Customer Count',
                    hue='Recommended Action', sizes=(20, 200), alpha=0.6, palette='viridis')
    plt.title('Churn Risk Clusters by Network Performance')
    plt.xlabel('Network Performance Score')
    plt.ylabel('Predicted Churn Risk')
    plt.legend(title='Recommended Action')
    plt.show()

# Helper function to create a geographic heatmap
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

# Helper function to create a scenario analysis chart
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

# Save the processed data for further analysis if needed
data_agg.to_csv('processed_telecom_churn_data.csv', index=False)
# ...existing code...

# Display the processed data as a table in the Streamlit app
st.write("## Telecom Churn Analysis Report")
st.dataframe(data_agg)

# Provide a download link for the CSV file
st.download_button(
    label="Download Report as CSV",
    data=data_agg.to_csv(index=False),
    file_name='processed_telecom_churn_data.csv',
    mime='text/csv'
)

# ...existing code...