from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nltk
import os
import re
import tarfile

from six.moves import urllib


DOWNLOADED_FILENAME = "../ImdbReviews.tar.gz"

def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)
    print("Found and verified IMDB Movie Reviews ")


download_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")

# Regex to only accept text and numbers
TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")


def get_reviews(dirname, positive=True):
    label = 1 if positive else 0

    reviews = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            with open(dirname + filename, "r+") as f:
                review = f.read().decode("utf-8")
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)

                reviews.append((review, label))
    return reviews


# Extracts the tar file
def extract_reviews():
    if not os.path.exists("aclImdb"):
       with tarfile.open(DOWNLOADED_FILENAME) as tar:
           tar.extractall()
           tar.close()

    positive_reviews = get_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews = get_reviews("aclImdb/train/neg/", positive=False)

    return positive_reviews, negative_reviews

# def get_review(path, positive=True):
#     label = 1 if positive else 0
#
#     with open(path, 'r') as f:
#         review_text = f.readlines()
#
#     reviews = []
#     for text in review_text:
#         reviews.append((text, label))
#
#     return reviews
#
#
# def extract_reviews():
#     positive_reviews = get_review("../rt-polarity.pos", positive=True)
#     negative_reviews = get_review("../rt-polarity.neg", positive=False)
#
#     return positive_reviews, negative_reviews


positive_reviews, negative_reviews = extract_reviews()

TRAIN_DATA = 5000
TOTAL_DATA = 6000

train_reviews = positive_reviews[:TRAIN_DATA] + negative_reviews[:TRAIN_DATA]

test_positive_reviews = positive_reviews[TRAIN_DATA:TOTAL_DATA]
test_negative_review = negative_reviews[TRAIN_DATA:TOTAL_DATA]


# Build training set
def get_vocabulary(train_reviews):
    words_set = set()

    for review in train_reviews:
        words_set.update(review[0].split())

    return list(words_set)


vocabulary = get_vocabulary(train_reviews)


def extract_features(review_text):
    # Split the review into words and create a set of words
    review_words = set(review_text.split())
    features = {}
    for word in vocabulary:
        features[word] = (word in review_words)
    return features


train_features = nltk.classify.apply_features(extract_features, train_reviews)

# Trained machine learning model
trained_classifier = nltk.NaiveBayesClassifier.train(train_features)


def sentiment_calculator(review_text):
    features = extract_features(review_text)
    return trained_classifier.classify(features)


def classify_test_review(test_positive_review, test_negative_review, sentiment_calculator):
    positive_results = [sentiment_calculator(review[0]) for review in test_positive_review]
    negative_results = [sentiment_calculator(review[0]) for review in test_negative_review]

    true_positives = sum(x > 0 for x in positive_results)
    true_negatives = sum(x == 0 for x in negative_results)

    percent_true_positive = float(true_positives) / len(positive_results)
    percent_true_negative = float(true_negatives) / len(negative_results)

    total_accurate = true_positives + true_negatives
    total = len(positive_results) + len(negative_results)

    print("Accuracy on positive reviews = " + "%.2f" % (percent_true_positive * 100) + "%")
    print("Accuracy on negative reviews = " + "%.2f" % (percent_true_negative * 100) + "%")
    print("Overall Accuracy = " + "%.2f" % (total_accurate * 100 / total) + "%")


classify_test_review(test_positive_reviews, test_positive_reviews, sentiment_calculator)