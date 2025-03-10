import pandas as pd
from src.data_loading import load_data
from src.data_processing import process_data
from src.calculated_metrics import calculate_metrics
from src.visualizations import create_visualizations
from src.sentiment_analysis import perform_sentiment_analysis
from src.proactive_alerts import check_and_send_alerts

def main():
    # Load the data
    data = load_data('data/sample_data.csv')
    
    # Process the data
    processed_data = process_data(data)
    
    # Calculate metrics
    metrics = calculate_metrics(processed_data)
    
    # Create visualizations
    create_visualizations(processed_data)
    
    # Perform sentiment analysis
    perform_sentiment_analysis(processed_data)
    
    # Check for alerts and send notifications
    check_and_send_alerts(processed_data)

if __name__ == "__main__":
    main()