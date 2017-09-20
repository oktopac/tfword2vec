from unittest import TestCase
import word2vec

import tensorflow as tf

class TestWord2Vec(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWord2Vec, self).__init__(*args, **kwargs)
        self.LEN_TRAINING = 100
        self.vocab = [str(i) for i in range(self.LEN_TRAINING)]

    def test_create_word2vec(self):
        with tf.Graph().as_default(), tf.Session() as session:
            w2v = word2vec.Word2Vec(session, self.vocab)



