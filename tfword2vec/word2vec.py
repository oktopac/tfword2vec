import logging
import numpy as np
import tensorflow as tf
import math
import os

class Word2Vec(object):
    def __init__(self, session, vocabulary, save_path=None):
        self.save_path = save_path
        self._session = session

        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)

        logging.info("Using %d vocab size" % self.vocabulary_size)
        self.batch_size = 256
        self.embedding_size = 128  # Dimension of the embedding vector.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.num_sampled = 64  # Number of negative examples to sample.

        self.build_graph()
        self.init.run()

    def forward(self):
        # Look up embeddings for inputs.
        self.embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_input)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        nce_loss = tf.nn.nce_loss(weights=nce_weights,
                                  biases=nce_biases,
                                  labels=self.train_labels,
                                  inputs=embed,
                                  num_sampled=self.num_sampled,
                                  num_classes=self.vocabulary_size)

        self.loss = tf.reduce_mean(nce_loss)
        tf.summary.scalar("Loss", self.loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

    def build_graph(self):
        # Input data.
        self.train_input = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        self.forward()

        self.similarity()

        self.saver = tf.train.Saver()

        # Add variable initializer.
        self.init = tf.global_variables_initializer()

    # Build the graph component that create the simliarity matrix for all embeddings
    def similarity(self):
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)
        self.similarity = tf.matmul(
            valid_embeddings, self.normalized_embeddings, transpose_b=True)

    # Evaluate the model, and save a checkpoint if its looking good
    def eval(self, n_batches, generate_batch):
        average_loss = 0
        for i in xrange(n_batches):
            batch_inputs, batch_labels = generate_batch(self.batch_size)
            feed_dict = {self.train_input: batch_inputs, self.train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            average_loss += self._session.run(self.loss, feed_dict=feed_dict)

        average_loss = 1.0 * average_loss / n_batches
        logging.info("Average loss on eval is %f" % average_loss)

        if self.save_path is not None:
            if average_loss < self.best_loss:
                logging.info("Saving best model with loss %f" % average_loss)

                self.saver.save(self._session,
                                os.path.join(self.save_path, "best_model.ckpt")
                                )

                self.best_loss = average_loss
            else:
                logging.info("Current loss (%f) worse than best (%f)" % (average_loss, self.best_loss))

    def train(self, num_steps, generate_batch):
        #TODO: fix this
        self.best_loss = 2**32

        graph = tf.Graph()

        # We must initialize all variables before we use them.
        # TODO: This check doesn't work on ml-engine
        print('Initialized, training for %d steps' % num_steps)
        if self.save_path:
            if tf.train.checkpoint_exists(self.save_path):
                logging.info("getting checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.save_path)
                if checkpoint:
                    logging.info("Restoring checkpoint from %s" % self.save_path)
                    self.saver.restore(self._session, checkpoint)
                else:
                    logging.info("No checkpoint to restore")

            # Sort out the summary stuff
            self.summary_writer = tf.summary.FileWriter(self.save_path, self._session.graph)

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(self.batch_size)
            feed_dict = {self.train_input: batch_inputs, self.train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = self._session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                self.eval(100, generate_batch)
                average_loss = 0

                if self.save_path is not None:
                    # Also use the summary writer
                    summary_op = tf.summary.merge_all()
                    summary_str = self._session.run(summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, step)

                    self.saver.save(self._session,
                                     os.path.join(self.save_path, "model.ckpt"),
                                     global_step=step)

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 100000 == 0:
                sim = self.similarity.eval()
                for i in xrange(self.valid_size):
                    valid_word = self.vocabulary[self.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = self.vocabulary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
