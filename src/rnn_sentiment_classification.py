from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import tarfile
import re
from six.moves import urllib

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import sys

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream




DOWNLOADED_FILENAME = "../ImdbReviews.tar.gz"
# todo find average tweet length
MAX_SEQUENCE_LENGTH = 250 # The number of of words to consider in each review, extend shorter reviews truncate longer reviews

# Regex to only accept text and numbers
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

# Init Twitter API
print("Initializing Twitter API")


# Variables that contains the user credentials to access Twitter API
access_token = "480819238-CQ95qCrHjEXdDzyEGnIZFMXoox4OtX3I0FFg9pBp"
access_token_secret = "JJWIb2Yio64C0hNCwKa8KNaNoqEoxzn02Tyx4LxVY2OVN"
consumer_key = "zNKTkmUiKMwwp3zdUA8daiVHX"
consumer_secret = "ORvVnkv7Y3d9KotlYKpXz4W5Y2Wqx9mWc7U7O0S2ymiIV2aAeA"


# This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    tweets = []
    count = 0

    def on_data(self, data):
        if self.count == 5815:
            sys.exit()

        if "text" in json.loads(data):
            self.tweets.append(json.loads(data)["text"])

        # Write to disk in batches of 100
        if len(self.tweets) % 100 == 0:
            print("Writing tweets to disk")
            self.tweet_to_disk(self.tweets)
            self.tweets = []
            self.count += 100

        return True

    def on_error(self, status):
        print(status)

    # For each read tweet write to disk
    @staticmethod
    def tweet_to_disk(tweets):
        filtered_tweets = []

        # Filter all the tweets
        for tweet in tweets:
            tweet = tweet.lower().replace("<br />", " ")
            tweet = tweet.lower().replace("&amp;", " ")
            tweet = tweet.lower().replace("rt", "")
            tweet = tweet.strip(' \t\n\r')
            tweet = re.sub(TOKEN_REGEX, '', tweet)

            if "http" in tweet:
                tweet = tweet[:tweet.index("http")]

            if len(tweet) > 0:
                filtered_tweets.append(tweet)

        # Write text to disk
        with open("./political/tweets.csv", "a") as text_file:
            for tweet in filtered_tweets:
                print("Writing tweet -> " + tweet)
                text_file.write(tweet + ",\n")

        print("Written to Disk")


# l = StdOutListener()
# auth = OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# stream = Stream(auth, l)

# This line filter Twitter Streams to capture data by the keywords passed into track
# Political ['trump', 'clinton', 'obama', 'tax', 'parkland', 'gun control', 'senate', 'rubio']
# stream.filter(track=['trump', 'clinton', 'obama', 'tax', 'parkland', 'gun control', 'senate', 'rubio'])


def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)
    print("Found and verified IMDB Movie Reviews ")


def get_reviews(dirname, positive=True):
    label = 1 if positive else 0

    reviews = []
    labels2 = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            with open(dirname + filename, "r+") as f:
                review = f.read().decode("utf-8")
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)

                reviews.append(review)
                labels2.append(label)
    return reviews, labels2


def get_tweets(political=True):
    file_path = ""
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


def extract_twitter_data():
    political_tweets, political_labels = get_tweets(political=True)
    non_political_tweets, non_political_labels = get_tweets(political=False)

    data_ret = political_tweets + non_political_tweets
    labels_ret = political_labels + non_political_labels

    return data_ret, labels_ret


def extract_labels_data():
    # if the file hasnt been extracted yet
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            tar.extractall()
            tar.close()

    positive_reviews, positive_labels = get_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews, negative_labels = get_reviews("aclImdb/train/neg/", positive=False)

    data2 = positive_reviews + negative_reviews
    labels3 = positive_labels + negative_labels

    return labels3, data2


print("Extracting Twitter Data")
# download_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
labels, data = extract_twitter_data()

# Map each word in dataset to unique numeric identifier (Truncates and pads documents of less than or greater than 250)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)

# Convert each review to its numeric representation
x_data = np.array(list(vocab_processor.fit_transform(data)))
y_output = np.array(labels)


np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]


# Increasing training data set to improve accuracy
TRAIN_DATA = 5000
TOTAL_DATA = 6000

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]

# Reset TF Graph and setup placeholders
tf.reset_default_graph()

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])  # Reviews
y = tf.placeholder(tf.int32, [None])  # Labels

# Setup NN

# Hyper parameters
num_epochs = 20  # very small
batch_size = 25 # 25 elements per batch
embedding_size = 50
max_label = 2  # Only possible values are 0,1 so max label = 2
vocabulary_size = len(vocab_processor.vocabulary_)

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

# Create the LSTM Cell todo this may have changed in newer TF version
lstmCell = tf.contrib.rnn.BasicLSTMCell(embedding_size)

# Prevent overfitting the model
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

# .75 implies that a neuron has a 75% change of being retained vs turned off so each time neurons are turned off and
# it forces other neurons to learn in their stead

# Unrolls the RNN Through time
_, (encoding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

# encoding is fed into softmax prediction layer

logits = tf.layers.dense(encoding, max_label, activation=None)

# calculate loss function softmax
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

loss = tf.reduce_mean(cross_entropy)

# Finally classify the review with the highest probability
prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.01) # Momentum based optimizer (gains momentum as it descends faster down the slope)
train_step = optimizer.minimize(loss)

# Training the NN
init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()

    for epoch in range(num_epochs):
        num_batches = int(len(train_data) // batch_size) + 1  # Floor division rounds to the nearest whole number

        for i in range(num_batches):
            min_ix = i * batch_size
            max_ix = np.min([len(train_data), ((i + 1) * batch_size)])

            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y: y_train_batch}
            session.run(train_step, feed_dict=train_dict)

            train_loss, train_acc = session.run([loss, accuracy], feed_dict=train_dict)

    test_dict = {x: test_data, y: test_target}
    test_loss, test_acc = session.run([loss, accuracy], feed_dict=test_dict)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_loss, test_acc))


