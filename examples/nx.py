import networkx as nx
import numpy as np
import itertools
import argparse
from tfword2vec import utils, word2vec
import logging
import tensorflow as tf
import os
aparser = argparse.ArgumentParser()

aparser.add_argument("output_directory")
aparser.add_argument("--vocab_size", type=int, default=1000)
aparser.add_argument("--context_window", type=int, default=10)
aparser.add_argument("--epochs", type=int, default=1)
aparser.add_argument("--learning_rate", type=float, default=1.0)

args = aparser.parse_args()

G=nx.fast_gnp_random_graph(1000, 0.1)

vocab = [str(i) for i in range(len(G))]
print(len(vocab))
source = []
target = []

for node in G.nodes_iter(data=False):
    nodes = list(G[node].keys())
    pairs = itertools.permutations(nodes, 2)
    for s, t in pairs:
        source.append(s)
        target.append(t)

features = np.array(source)
labels = np.array(target)

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

dataset = dataset.repeat(1)
dataset = dataset.batch(10)

iterator = dataset.make_initializable_iterator()
with tf.Session() as session:
    next_element = iterator.get_next()
    session.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

    for i in range(10)
    value = session.run(next_element)
    print(value)
#
# with tf.Graph().as_default(), tf.Session() as session:

    # w2v = word2vec.Word2Vec(session, vocab, args.output_directory, learning_rate=args.learning_rate)
    # w2v.train(args.epochs, lambda: generator())
