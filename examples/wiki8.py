from tfword2vec import utils, word2vec
import argparse
import logging
import tensorflow as tf
import os

aparser = argparse.ArgumentParser()

aparser.add_argument("document_file")
aparser.add_argument("output_directory")
aparser.add_argument("--vocab_size", type=int, default=1000)
aparser.add_argument("--context_window", type=int, default=10)
aparser.add_argument("--epochs", type=int, default=1)
aparser.add_argument("--learning_rate", type=float, default=1.0)

args = aparser.parse_args()

logging.basicConfig(level=logging.INFO)

logging.info("Got document file %s" % args.document_file)

if not os.path.exists(args.output_directory):
    logging.info("Creating output directory")
    os.makedirs(args.output_directory)

generate_single, vocab = utils.setup_document_generator(args.document_file, args.output_directory,
                                                        args.vocab_size, args.context_window)


with tf.Graph().as_default(), tf.Session() as session:
    w2v = word2vec.Word2Vec(session, vocab, args.output_directory, learning_rate=args.learning_rate)
    w2v.train(args.epochs, generate_single)