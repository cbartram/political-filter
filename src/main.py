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

VOCABULARY_SIZE = 5000
URL_PATH = "http://mattmahoney.net/dc/text8.zip"
FILESIZE = 31344016

maybe_download(URL_PATH, FILESIZE)

# 17 Million words in our dataset
vocabulary = read_words()

word_counts, word_indexes, dictionary, reversed_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)




