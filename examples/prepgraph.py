import networkx as nx
import numpy as np
import itertools
import argparse
from tfword2vec import utils, word2vec
import logging
import tensorflow as tf
import os
import operator

aparser = argparse.ArgumentParser()

logging.basicConfig(level=logging.INFO)

aparser.add_argument("graphfile")
aparser.add_argument("prout")

args = aparser.parse_args()

logging.info("Reading edge list")
G = nx.read_edgelist(args.graphfile)

logging.info("Running page rank")
# pr = nx.pagerank(G, max_iter=1)
pr = nx.degree(G)

sorted_pr = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)

with open(args.prout, 'w') as fd:
    for k, v in sorted_pr:
        fd.write("%s,%d\n" % (k, v))