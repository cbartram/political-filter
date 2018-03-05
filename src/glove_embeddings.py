from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from src.TweetParser import TweetParser

MAX_SEQUENCE_LENGTH = 250 # The number of of words to consider in each review, extend shorter reviews truncate longer reviews

labels, data = TweetParser().extract_twitter_data()
words = np.load("../data/wordsList.npy")

def get_word_index_dictionary():
    dictionary = {}

    index = 0
    for word in words:
        dictionary[word] = index
        index += 1
    return dictionary

dictionary = get_word_index_dictionary()

review_ids = np.load('../data/idsMatrix.npy')

vocabulary = TweetParser().get_vocabulary()
vocabulary_size = 10000
embed = TweetParser.embed_tweets(vocabulary, vocabulary_size)

print(embed[0:1])

x_data = review_ids
y_output = np.array(labels)

np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]

# Increasing training data set to improve accuracy
TRAIN_DATA = 5000
TOTAL_DATA = 7000

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
vocabulary_size = len(words)

saved_embeddings = np.load("../data/wordVectors.npy")

# embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

embeddings = tf.nn.embedding_lookup(saved_embeddings, x)

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


