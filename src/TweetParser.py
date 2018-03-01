
"""
 TweetParser - This class reads tweets from disk and formats them (and labels) as an array
 @author cbartram
"""
class TweetParser:
    tweet = []

    def __init__(self):
        self.tweet = []

    def extract_twitter_data(self):
        political_tweets, political_labels = self.__get_tweets(political=True)
        non_political_tweets, non_political_labels = self.__get_tweets(political=False)

        data_ret = political_tweets + non_political_tweets
        labels_ret = political_labels + non_political_labels

        return labels_ret, data_ret

    """
     Private method returns the tweets and their respective labels 
     as lists
    """
    @staticmethod
    def __get_tweets(political=True):
        tweets = []  # The tweet text
        labels_features = []  # The label text (is or isnt political)

        label = 1 if political else 0

        if political:
            file_path = "./political/tweets.csv"
        else:
            file_path = "./non_political/tweets.csv"

        #  Open file and read lines
        with open(file_path, "r+") as f:
            for line in f:
                tweets.append(line)
                labels_features.append(label)

        return tweets, labels_features


