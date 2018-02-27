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


DOWNLOADED_FILENAME = "../ImdbReviews.tar.gz"
# todo find average tweet length
MAX_SEQUENCE_LENGTH = 250 # The number of of words to consider in each review, extend shorter reviews truncate longer reviews

# Regex to only accept text and numbers
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")


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

# Map each word in dataset to unique numeric identifier (TRUNCATES and pads documents)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)

# Convert each review to its numeric representation
x_data = np.array(list(vocab_processor.fit_transform(data)))
y_output = np.array(labels)








