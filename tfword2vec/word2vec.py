import logging
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.client import timeline

class Word2Vec(object):
    def __init__(self, session, vocabulary=None, save_path=None, learning_rate=1.0):
        self._test_data = False
        self.save_path = save_path
        self.session = session
        self.learning_rate = learning_rate

        self.features_placeholder = tf.placeholder(np.int64)
        self.labels_placeholder = tf.placeholder(np.int64)

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

    def add_test_data(self, features, labels):
        self._test_data = True
        self._test_features = features
        self._test_labels = labels

    def add_training_data(self, features, labels):
        self._features = features
        self._labels = labels

    def initialise_training_iterator(self):
        self.session.run(self.train_iterator.initializer,
                         feed_dict={
                            self.features_placeholder: self._features,
                            self.labels_placeholder: self._labels
                         })


    def forward(self):
        # Look up embeddings for inputs.
        with tf.name_scope("embed"):
            self.embedding_matrix = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="embedding_matrix"
            )

        with  tf.name_scope('loss'):
            # This takes the full embedding matrix, and pulls out only the rows that are required for this example (i.e. the
            # rows that are referenced in the train_input)
            reduced_embed_matrix = tf.nn.embedding_lookup(self.embedding_matrix, self.train_input)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / self.embedding_size ** 0.5))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            nce_loss = tf.nn.nce_loss(weights=nce_weights,
                                      biases=nce_biases,
                                      labels=self.train_labels,
                                      inputs=reduced_embed_matrix,
                                      num_sampled=self.num_sampled,
                                      num_classes=self.vocabulary_size)
        with  tf.name_scope('optimize'):
            self.loss = tf.reduce_mean(nce_loss)

            # Construct the SGD optimizer using a learning rate of self.learning_rate
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.global_step)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        tf.summary.scalar("TrainLoss", self.loss)


    def build_graph(self):
        # Input data.
        with tf.name_scope("data"):
            self.dataset = tf.contrib.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))

            self.dataset = self.dataset.batch(self.batch_size).shuffle(buffer_size=10000)

            self.train_iterator = self.dataset.make_initializable_iterator()

            self.train_input, self.train_labels = self.train_iterator.get_next()

        self.forward()

        self.similarity()

        self.saver = tf.train.Saver()

        # Add variable initializer.
        self.init = tf.global_variables_initializer()

    # Build the graph component that create the simliarity matrix for all embeddings
    def similarity(self):
        with  tf.name_scope('similarity'):
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keep_dims=True))
            self.normalized_embeddings = self.embedding_matrix / norm
            valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(
                valid_embeddings, self.normalized_embeddings, transpose_b=True)

    # Evaluate the model, and save a checkpoint if its looking good
    def test(self):
        self.session.run(self.train_iterator.initializer,
                         feed_dict={
                            self.features_placeholder: self._test_features,
                            self.labels_placeholder: self._test_labels
                         })
        average_loss = 0
        n_batches = 1

        try:
            while True:
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                average_loss += self.session.run(self.loss)
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        average_loss = 1.0 * average_loss / n_batches
        logging.info("Average loss on eval is %f" % average_loss)
        return average_loss

    def train(self, num_epochs, trace=False):
        #TODO: fix this
        self.best_loss = 2**32

        graph = tf.Graph()

        # We must initialize all variables before we use them.
        # TODO: This check doesn't work on ml-engine
        print('Initialized, training for %d epochs' % num_epochs)

        if self.save_path:
            if tf.train.checkpoint_exists(self.save_path):
                logging.info("getting checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.save_path)
                if checkpoint:
                    logging.info("Restoring checkpoint from %s" % self.save_path)
                    self.saver.restore(self.session, checkpoint)
                else:
                    logging.info("No checkpoint to restore")

            # Sort out the summary stuff
            self.summary_writer = tf.summary.FileWriter(self.save_path, self.session.graph)



        for epoch in range(num_epochs):
            logging.info("Running on epoch %d" % epoch)
            average_loss = 0
            self.initialise_training_iterator()
            try:
                i = 0
                while True:
                    if trace and i == 100:
                        run_metadata = tf.RunMetadata()
                        _, loss_val = self.session.run([self.optimizer, self.loss],
                               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                               run_metadata=run_metadata)
                        average_loss += loss_val

                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        with open("%s/trace.json" % self.save_path, 'w') as trace_file:
                            trace_file.write(trace.generate_chrome_trace_format())

                    i += 1
                    # We perform one update step by evaluating the optimizer op (including it
                    # in the list of returned values for session.run()


                    if i % 1000 == 0 and self.save_path is not None:
                        # Also use the summary writer
                        summary_op = tf.summary.merge_all()
                        _, loss_val, summary_str = self.session.run([self.optimizer, self.loss, summary_op])
                        average_loss += loss_val

                        global_step = tf.train.global_step(self.session, self.global_step)

                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at global_step %d epoch %d: %f' % (
                        global_step, epoch, average_loss / 1000.))

                        self.summary_writer.add_summary(summary_str, global_step)

                        self.saver.save(self.session,
                                        os.path.join(self.save_path, "model.ckpt"),
                                        global_step=self.global_step)

                        average_loss = 0
                    else:
                        _, loss_val = self.session.run([self.optimizer, self.loss])
                        average_loss += loss_val

            except tf.errors.OutOfRangeError:
                pass

            if self._test_data:
                test_loss = self.test()

                if self.save_path is not None:
                    global_step = tf.train.global_step(self.session, self.global_step)

                    test_loss_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="TestLoss", simple_value=test_loss),
                    ])

                    self.summary_writer.add_summary(test_loss_summary, global_step)

                    if average_loss < self.best_loss:
                        logging.info("Saving best model with loss %f" % test_loss)

                        self.saver.save(self.session,
                                        os.path.join(self.save_path, "best_model.ckpt")
                                        )

                        self.best_loss = test_loss

                    else:
                        logging.info("Current loss (%f) worse than best (%f)" % (test_loss, self.best_loss))

            #
            # # Note that this is expensive (~20% slowdown if computed every 500 steps)
            # if batch_step % 100000 == 0:
            #     sim = self.similarity.eval()
            #     for i in range(self.valid_size):
            #         valid_word = self.vocabulary[self.valid_examples[i]]
            #         top_k = 8  # number of nearest neighbors
            #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #         log_str = 'Nearest to %s:' % valid_word
            #         for k in range(top_k):
            #             close_word = self.vocabulary[nearest[k]]
            #             log_str = '%s %s,' % (log_str, close_word)
            #         print(log_str)
