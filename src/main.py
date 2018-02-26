from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
from six.moves import urllib
from six.moves import xrange

import numpy as np
import tensorflow as tf

DOWNLOADED_FILENAME = '/Users/ilp281/PycharmProjects/political-filter/SampleText.zip'


def maybe_download(url_path, expected_bytes):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)

    statinfo = os.stat(DOWNLOADED_FILENAME)
    if statinfo.st_size == expected_bytes:
        print("Found and verified file from this path: ", url_path)
        print("Downloaded File: ", DOWNLOADED_FILENAME)
    else:
        print("File size does not match the expected size. File corrupted.")
        print(statinfo.st_size)
        raise Exception("Failed to verify file from " + url_path)


def read_words():
    with zipfile.ZipFile(DOWNLOADED_FILENAME) as f:
        firstfile = f.namelist()[0]
        # Convert contents into string format
        filestring = tf.compat.as_str(f.read(firstfile))
        words = filestring.split()
    return words


def build_dataset(words, n_words):
    word_counts = [['UNKNOWN', -1]]

    counter = collections.Counter(words)
    word_counts.extend(counter.most_common(n_words - 1))

    dictionary = dict()

    # Stor mapping of word to index variable
    for word, _ in word_counts:
        # Assign unique indeces to words; most common words have lowest index values
        # {"all": 253, "the": 3}
        dictionary[word] = len(dictionary)

    word_indexes = list()

    unknown_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 # Dictionary['unknown']
            unknown_count += 1

        word_indexes.append(index)
    word_counts[0][1] = unknown_count

    # {1: "is", 2356: "how", 43: "this"}
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return word_counts, word_indexes, dictionary, reversed_dictionary

"""
 Skip window is the window for how many words we should predict to the left & right of the 
 word index i.e. "The quick brown fox jumped over the lazy brown dog" with skip_window 3 and word_index fox 
 would yield all words between The -> the (+- 3 words from fox) 
 
 Num skips is the number of words we will use surrounding the target word
"""
def generate_batch(word_indexes, batch_size, num_skips, skip_window):
    global global_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [skip_window input_word skip_window]

    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)

    for i in range(batch_size // num_skips):
        target = skip_window # input word at the center of the buffer (deque)
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)

            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word "fox"
            labels[i * num_skips + j, 0] = buffer[target]  # these are the context words fox->quick fox->brown
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)

    global_index = (global_index + len(word_indexes) - span) % len(word_indexes)

    return batch, labels


# Execute methods defined above

VOCABULARY_SIZE = 5000
URL_PATH = "http://mattmahoney.net/dc/text8.zip"
FILESIZE = 31344016

maybe_download(URL_PATH, FILESIZE)

# 17 Million words in our dataset
vocabulary = read_words()

word_counts, word_indexes, dictionary, reversed_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)

# No longer need vocabulary
del vocabulary

global_index = 0

batch, labels = generate_batch(word_indexes, 10, 2, 5)

valid_size = 16
valid_window = 100

valid_examples = np.random.choice(valid_window, valid_size, replace=False)

batch_size = 128
embedding_size = 50
skip_window = 2
num_skips = 2


# Build fresh tensor graph
tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embedding_size], -1.0, 1.0))

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# NN Layer with no activation function (linear layer)
weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

train_one_hot = tf.one_hot(train_labels, VOCABULARY_SIZE)
# Determines probability if the output is true or false (2 probabilities)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out, labels=train_one_hot))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# L2 Norm is the sqrt of the sum of the squares of a vector
l2_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))

normalized_embeddings = embeddings / l2_norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Initialize and train
init = tf.global_variables_initializer()
num_steps = 20001 # num of epochs in training

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(word_indexes, batch_size, num_skips, skip_window)

        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Calculate avg loss every 2,000 steps
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000

            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Very compute heavy
        if step % 10000 == 0:
            sim = similarity.eval()

            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8 # Number of nearest neighbors

                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word

                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)


