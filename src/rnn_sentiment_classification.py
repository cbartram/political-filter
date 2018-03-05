from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import re
from six.moves import urllib

import numpy as np
import tensorflow as tf

from tweepy import OAuthHandler
from tweepy import Stream

# Local Packages
from src.TweetWriter import TweetWriter
from src.TweetParser import TweetParser


MAX_SEQUENCE_LENGTH = 50 # The number of of words to consider in each review, extend shorter reviews truncate longer reviews

# Regex to only accept text and numbers
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

# Init Twitter API
print("Initializing Twitter API")

# Variables that contains the user credentials to access Twitter API
access_token = os.environ.get('ACCESS_TOKEN')
access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')
consumer_key = os.environ.get('CONSUMER_KEY')
consumer_secret = os.environ.get('CONSUMER_SECRET')

# Init tweet writer
tweet_writer = TweetWriter()

# Set which file to write to (political or non political)
tweet_writer.set_tweet_category(True) # True = political False = non political

# auth = OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# stream = Stream(auth, tweet_writer)

# This line filter Twitter Streams to capture data by the keywords passed into track
# Political categories ['trump', 'clinton', 'obama', 'tax', 'parkland', 'gun control', 'senate', 'rubio']
# Non Political categories ['football', 'nfl', 'gains', 'gym', 'cars', 'fishing', 'painting', 'music', 'apple', 'iphone']
# stream.filter(track=['trump', 'clinton', 'obama', 'tax', 'parkland', 'gun control', 'senate', 'rubio'])

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


print("Extracting Twitter Data")
labels, data = TweetParser().extract_twitter_data()

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
TRAIN_DATA = 15000
TOTAL_DATA = 17000

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
num_epochs = 25  # very small
batch_size = 25 # 25 elements per batch
embedding_size = 50
max_label = 2  # Only possible values are 0,1 so max label = 2
vocabulary_size = len(vocab_processor.vocabulary_)

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

# Create the LSTM Cell
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

optimizer = tf.train.AdamOptimizer(0.01) # Momentum based optimizer gains momentum as it descends faster down the slope
train_step = optimizer.minimize(loss)

# Training the NN
init = tf.global_variables_initializer()

# Save the model
saver = tf.train.Saver()

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

    save_path = saver.save(session, "../checkpoint/ltsm_rnn.ckpt")
    print("Model saved in path: %s" % save_path)

