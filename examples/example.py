import numpy as np
import argparse
from tfword2vec import word2vec
import logging
import tensorflow as tf
import os
import uuid
from sklearn import model_selection

aparser = argparse.ArgumentParser()

aparser.add_argument("--epochs", type=int, default=100)
aparser.add_argument("--learning_rate", type=float, default=1.0)

args = aparser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

features = np.random.randint(0, 1000, 10000000)
labels = np.random.randint(0, 1000, 10000000)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features, labels, test_size=0.01, random_state=42)

y_train = y_train.reshape((len(y_train), 1))
y_test = y_test.reshape((len(y_test), 1))

output_directory = "/tmp/testing_tfword2vec/%s" % str(uuid.uuid4())[-8:]

logging.info("Using output directory %s" % output_directory)

os.makedirs(output_directory)

logging.info("Training model")

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Graph().as_default(), tf.Session(config=config) as session:

    w2v = word2vec.Word2Vec(session, range(1000), output_directory)
    w2v.add_training_data(X_train, y_train)
    w2v.add_test_data(X_test, y_test)
    w2v.train(args.epochs, trace="trace.json")