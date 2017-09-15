import collections
import random
import numpy as np
import tensorflow as tf

def read_words(fname):
    with open(fname, 'r') as fd:
        words = tf.compat.as_str(fd.read()).split()

    return words

def generate_vocab(words, n=None):
    vocab = collections.Counter(words)

    return vocab.most_common(n=n)

def save_vocab(vocab, path):
    with open(path, 'w') as fd:
        for word, count in vocab:
            fd.write("%s,%d\n" % (word, count))
    fd.close()

def generate_w2i_lookup(vocab):
    assert(type(vocab)) == list
    w2i = {}
    for i, (word, count) in enumerate(vocab):
        w2i[word] = i

    return w2i

def generate_i2w_lookup(vocab):
    return [x[0] for x in vocab]

def generate_index_document(w2i, words):
    return [w2i[word] for word in words if word in w2i]

def generate_sample(index_document, context_window_size):
    for index, center in enumerate(index_document):
        context = random.randint(1, context_window_size)

        # get a random target before the center word
        for target in index_document[max(0, index - context): index]:
            yield center, target

        # get a random target after the center wrod
        for target in index_document[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def setup_document_generator(fname, outdir, vocab_size, skip_window):
    words = read_words(fname)
    vocab = generate_vocab(words, vocab_size)
    save_vocab(vocab, "%s/vocab.csv" % outdir)
    w2i = generate_w2i_lookup(vocab)
    i2w = generate_i2w_lookup(vocab)
    index_document = generate_index_document(w2i, words)
    del words
    # Here we create a function that starts a new single generator
    single_generator = lambda: generate_sample(index_document, skip_window)
    return single_generator, i2w