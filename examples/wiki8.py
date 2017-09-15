from tfword2vec import utils, word2vec
import argparse
import logging
import tensorflow as tf

aparser = argparse.ArgumentParser()

aparser.add_argument("document_file")
aparser.add_argument("output_directory")
aparser.add_argument("--vocab_size", type=int, default=1000)
aparser.add_argument("--context_window", type=int, default=10)
aparser.add_argument("--epochs", type=int, default=10)

args = aparser.parse_args()

logging.basicConfig(level=logging.INFO)

logging.info("Got document file %s" % args.document_file)

generate_single, vocab = utils.setup_document_generator(args.document_file, args.output_directory,
                                                        args.vocab_size, args.context_window)

with tf.Graph().as_default(), tf.Session() as session:
    w2v = word2vec.Word2Vec(session, vocab, args.output_directory)
    w2v.train(args.epochs, generate_single)