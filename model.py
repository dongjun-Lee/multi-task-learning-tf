import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, vocabulary_size, num_class, args):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden

        self.x = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.lm_y = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.clf_y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(embeddings, self.x)

        with tf.name_scope("rnn"):
            cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)

        with tf.name_scope("lm-output"):
            self.lm_logits = tf.layers.dense(rnn_outputs, vocabulary_size)

        with tf.name_scope("clf-output"):
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.clf_logits = tf.layers.dense(rnn_outputs_flat, num_class)
            self.clf_predictions = tf.argmax(self.clf_logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.lm_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.lm_logits,
                targets=self.lm_y,
                weights=tf.sequence_mask(self.x_len, args.max_document_len, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
            self.clf_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clf_logits, labels=self.clf_y))
            self.total_loss = self.lm_loss + self.clf_loss

        with tf.name_scope("clf-accuracy"):
            correct_predictions = tf.equal(self.clf_predictions, self.clf_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def make_cell(self):
        cell = rnn.BasicLSTMCell(self.num_hidden)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        return cell
