import tweepy
from textblob import TextBlob
import pandas as pd

def authenticate_twitter_api(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def get_tweets(api, query, count=100):
    tweets = []
    try:
        fetched_tweets = api.search(q=query, count=count)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet['text'] = tweet.text
            parsed_tweet['sentiment'] = TextBlob(tweet.text).sentiment.polarity
            tweets.append(parsed_tweet)
    except tweepy.TweepError as e:
        print(f"Error: {str(e)}")
    return tweets

def analyze_sentiment(data, consumer_key, consumer_secret, access_token, access_token_secret):
    api = authenticate_twitter_api(consumer_key, consumer_secret, access_token, access_token_secret)
    
    sentiment_scores = []
    for index, row in data.iterrows():
        tweets = get_tweets(api, query=row['Location'] + ' cellular reception', count=100)
        if tweets:
            avg_sentiment = pd.Series([tweet['sentiment'] for tweet in tweets]).mean()
            sentiment_scores.append(avg_sentiment)
        else:
            sentiment_scores.append(0)
    
    data['Sentiment Score'] = sentiment_scores
    return data