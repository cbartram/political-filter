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
import twitter
import json




DOWNLOADED_FILENAME = "../ImdbReviews.tar.gz"
# todo find average tweet length
MAX_SEQUENCE_LENGTH = 250 # The number of of words to consider in each review, extend shorter reviews truncate longer reviews

# Regex to only accept text and numbers
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

# Init Twitter API
print("Initializing Twitter API")
api = twitter.Api(consumer_key="zNKTkmUiKMwwp3zdUA8daiVHX",
                  consumer_secret="ORvVnkv7Y3d9KotlYKpXz4W5Y2Wqx9mWc7U7O0S2ymiIV2aAeA",
                  access_token_key="480819238-CQ95qCrHjEXdDzyEGnIZFMXoox4OtX3I0FFg9pBp",
                  access_token_secret="JJWIb2Yio64C0hNCwKa8KNaNoqEoxzn02Tyx4LxVY2OVN")

results = api.GetSearch(
    raw_query="q=twitter%20&result_type=recent&since=2014-07-19&count=100")

dump = json.dumps(results[99]._json)
load = json.loads(dump)


# For each read tweet write to disk
def tweet_to_disk(api_response):
    for tweet in api_response:
        tweet = tweet._json
        loaded_tweet = json.loads(tweet._json)
        # print(loaded_tweet["text"])



tweet_to_disk(results)


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

download_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
labels, data = extract_labels_data()

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


