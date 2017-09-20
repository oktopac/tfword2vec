import networkx as nx
import numpy as np
import itertools
import argparse
from tfword2vec import utils, word2vec
import logging
import tensorflow as tf
import os
import operator
from sklearn import model_selection

aparser = argparse.ArgumentParser()

logging.basicConfig(level=logging.INFO)

aparser.add_argument("graphfile")
aparser.add_argument("vocablist")
aparser.add_argument("output_directory")
aparser.add_argument("--vocab_size", type=int, default=1000)
aparser.add_argument("--context_window", type=int, default=10)
aparser.add_argument("--epochs", type=int, default=1)
aparser.add_argument("--learning_rate", type=float, default=1.0)

args = aparser.parse_args()

logging.info("Reading edge list")
G = nx.read_edgelist(args.graphfile)

logging.info("Reading page rank")
vocab = []
vocab_to_id = {}
with open(args.vocablist, 'r') as fd:
    for i in range(args.vocab_size):
        node_id = fd.readline().split(',')[0]
        vocab.append(node_id)
        vocab_to_id[node_id] = i

source = []
target = []

logging.info("Generating pairs of context")
for node in G.nodes_iter(data=False):
    nodes = list(G[node].keys())
    valid_nodes = filter(lambda node: node in vocab_to_id, nodes)
    pairs = itertools.permutations(valid_nodes, 2)
    for s, t in pairs:
        source.append(vocab_to_id[s])
        target.append(vocab_to_id[t])

logging.info("Generating Test/Train Split")
features = np.array(source)
labels = np.array(target)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features, labels, test_size=0.01, random_state=42)

print(X_train.shape)
print(X_test.shape)
y_train = y_train.reshape((len(y_train), 1))
y_test = y_test.reshape((len(y_test), 1))
print(y_train.shape)
print(y_test.shape)

if not os.path.exists(args.output_directory):
    logging.info("Creating output directory")
    os.makedirs(args.output_directory)

logging.info("Training model")


with tf.Graph().as_default(), tf.Session() as session:

    w2v = word2vec.Word2Vec(session, vocab, args.output_directory, learning_rate=args.learning_rate)
    w2v.add_training_data(X_train, y_train)
    w2v.add_test_data(X_test, y_test)
    w2v.train(args.epochs, trace="trace.json")