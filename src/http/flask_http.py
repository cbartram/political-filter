# Imports
from flask import Flask
from flask import request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route("/api/v1/feed", methods=['POST'])
def feed():
    sentences = request.get_json(silent=True).get("sentences")

    # Map each word in dataset to unique numeric identifier (Truncates and pads documents of less than or greater than 250)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(50)

    # Convert each review to its numeric representation
    x_data = np.array(list(vocab_processor.fit_transform(sentences)))
    y_output = np.array([1] * len(sentences)) # An array of 1's [1, 1, 1, ,1]

    x = tf.placeholder(tf.int32, [None, 50])  # Reviews
    y = tf.placeholder(tf.int32, [None])  # Labels

    # Setup NN

    # Hyper parameters
    embedding_size = 50
    max_label = 2  # Only possible values are 0,1 so max label = 2
    vocabulary_size = len(vocab_processor.vocabulary_)

    embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

    # Create the LSTM Cell
    lstmCell = tf.contrib.rnn.BasicLSTMCell(embedding_size)

    # Prevent overfitting the model
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

    # Unrolls the RNN Through time
    _, (encoding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

    # encoding is fed into softmax prediction layer
    logits = tf.layers.dense(encoding, max_label, activation=None)

    # Finally classify the review with the highest probability
    prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, "../../checkpoint/ltsm_rnn.ckpt")

        test_dict = {x: x_data, y: y_output}
        p = session.run([prediction], feed_dict=test_dict)
        print(p)
        for i, s in enumerate(sentences):
            print("%s -> " % s)
            print(p)

    return "success!"