import sys
import json
import preprocessor as p
from langdetect import detect
from tweepy.streaming import StreamListener

"""
Tweet Writer - Preprocess's tweets for valid text and writes them to disk
@author Cbartram
"""
class TweetWriter(StreamListener):
    tweets = []
    count = 0
    count_threshold = 10000
    political = True

    def __init__(self):
        self.tweets = []
        self.count = 0
        self.count_threshold = 10000  # Defaults to 100000
        self.political = True

    def set_count_thresh(self, count_threshold):
        if count_threshold > 0:
            self.count_threshold = count_threshold
        else:
            print("Threshold must be a 32bit integer Greater than zero.")

    def set_tweet_category(self, tweet_category):
        self.political = tweet_category

    def on_data(self, data):
        if self.count == self.count_threshold:
            sys.exit()

        if "text" in json.loads(data):
            self.tweets.append(json.loads(data)["text"])

        # Write to disk in batches of 100
        if len(self.tweets) % 100 == 0:
            print("Writing tweets to disk")
            self.tweet_to_disk(self.tweets, self.political)
            self.tweets = []
            self.count += 100

        return True

    def on_error(self, status):
        print(status)

    # For each read tweet write to disk
    @staticmethod
    def tweet_to_disk(tweets, political=True):
        filtered_tweets = []

        if political:
            path = "./political/tweets.csv"
        else:
            path = "./non_political/tweets.csv"

            # Filter all the tweets
        for tweet in tweets:
            # Filter out language and remove tweeters username
            try:
                # Preprocess tweet
                tweet = p.clean(tweet)

                language = detect(tweet)
                tweet = tweet[tweet.index(" "):len(tweet)]
                tweet = tweet.rstrip(' \t\n\r')
                tweet = tweet.lstrip(' \t\n\r')

                if len(tweet) > 30 and language == "en":
                    filtered_tweets.append(tweet)
            except:
                print("Could not parse tweet:", tweet)

        # Write text to disk
        with open(path, "a") as text_file:
            for tweet in filtered_tweets:
                print("Writing tweet -> " + tweet)
                text_file.write(tweet + "\n")

        print("Tweets Written to Disk")
