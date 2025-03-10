# Telecom Churn Analysis Report

## Overview
The Telecom Churn Analysis project aims to analyze customer churn in the telecommunications industry by leveraging data processing, calculated metrics, visualizations, social media sentiment analysis, and proactive alerts. This project provides insights into customer behavior and network performance, enabling stakeholders to make informed decisions.

## Project Structure
```
telecom-churn-analysis
├── data
│   └── sample_data.csv
├── src
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── calculated_metrics.py
│   ├── visualizations.py
│   ├── sentiment_analysis.py
│   ├── proactive_alerts.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Data
The dataset used in this project is located in the `data/sample_data.csv` file. It contains the following columns:
- **Location**: The city where the customers are located.
- **Latitude**: The geographical latitude of the location.
- **Longitude**: The geographical longitude of the location.
- **Customer Count**: The number of customers in that location.
- **Avg Monthly Revenue**: The average revenue generated per customer per month.
- **Network Performance Score**: A score representing the performance of the network in that area.
- **Predicted Churn Risk**: The likelihood of customers churning in that location.
- **Recommended Action**: Suggested actions based on churn risk.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd telecom-churn-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
To run the analysis, execute the main script:
```
python src/main.py
```
This will orchestrate the loading of data, processing, calculation of metrics, generation of visualizations, sentiment analysis, and sending of proactive alerts.

## Features
- **Data Loading**: Load the dataset into a pandas DataFrame for analysis.
- **Data Processing**: Clean and aggregate the data to prepare it for analysis.
- **Calculated Metrics**: Compute key metrics such as Revenue at Risk, Revenue Protected, and ROI.
- **Visualizations**: Create various visualizations to represent the data and insights effectively.
- **Social Media Sentiment Analysis**: Analyze social media sentiment related to cellular reception using the Twitter API.
- **Proactive Alerts**: Send automated alerts based on network performance and churn risk.

## Conclusion
This project provides a comprehensive analysis of customer churn in the telecommunications sector, utilizing data-driven insights to enhance decision-making processes.