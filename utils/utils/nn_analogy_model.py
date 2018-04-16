
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

def multi_batch_generator(batch_size, *data_arrays):
    """Generate minibatches from multiple columns of data.
    Example:
        for (bx, by) in multi_batch_generator(5, x, y):
            # bx is minibatch for x
            # by is minibatch for y
    Args:
      batch_size: int, batch size
      data_arrays: one or more array-like, supporting slicing along the first
        dimension, and with matching first dimension.
    Yields:
      minibatches of maximum size batch_size
    """
    assert(data_arrays)
    num_examples = len(data_arrays[0])
    for i in range(1, len(data_arrays)):
        assert(len(data_arrays[i]) == num_examples)

    for i in range(0, num_examples, batch_size):
        # Yield matching slices from each data array.
        yield tuple(data[i:i+batch_size] for data in data_arrays)

def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class nn_analogy_model(object):
    
    id_to_word = {}
    word_to_id = {}
    vocab = []
    embed = []

    def __init__(self, embed_file, graph=None, *args, **kwargs):
        """Init function.
        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.
        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        print("Reading embedding file...")
        self.readEmbedFile(embed_file)

    def readEmbedFile(self, filename):
        embed_file = open(filename, 'r')
        for line in embed_file.readlines()
            row = line.strip().split()
            self.id_to_word[len(vocab)] = row[0]
            self.word_to_id[row[0]] = len(vocab)
            self.vocab.append(row[0])
            self.embed.append(row[1:])
        embed_file.close()

    @with_self_graph
    def embedding_layer(self):
        """Construct an embedding layer.
        You should define a variable for the embedding matrix, and initialize it
        using tf.random_uniform_initializer to values in [-init_scale, init_scale].
        Hint: use tf.nn.embedding_lookup
        Args:
            ids_: [batch_size, max_len] Tensor of int32, integer ids
            V: (int) vocabulary size
            embed_dim: (int) embedding dimension
            init_scale: (float) scale to initialize embeddings
        Returns:
            xs_: [batch_size, max_len, embed_dim] Tensor of float32, embeddings for
                each element in ids_
        """
        with tf.name_scope("Embedding_Layer"):
            V_size = len(self.vocab)
            embed_dim = len(self.embed[0]) 
            W_embed_ = tf.get_variable("W_embed",shape=[V_size, embed_dim],trainable=False).assign(np.asarray(self.embed))
         return W_embed_

    @with_self_graph
    def fully_connected_layers(self, h0_, hidden_dims, activation=tf.tanh,
                               dropout_rate=0,is_training):
        """Construct a stack of fully-connected layers.
        This is almost identical to the implementation from A1, except that we use
        tf.layers.dense for convenience.
        Args:
            h0_: [batch_size, d] Tensor of float32, the input activations
            hidden_dims: list(int) dimensions of the output of each layer
            activation: TensorFlow function, such as tf.tanh. Passed to
                tf.layers.dense.
            dropout_rate: if > 0, will apply dropout to activations.
            is_training: (bool) if true, is in training mode
        Returns:
            h_: [batch_size, hidden_dims[-1]] Tensor of float32, the activations of
                the last layer constructed by this function.
        """
        with tf.name_scope("Deep_Layer"):
            h_ = h0_
            for i, hdim in enumerate(hidden_dims):
                h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))
                if dropout_rate > 0:
                    h_ = tf.layers.dropout(h_,rate=dropout_rate,training=is_training)

        return h_

    @with_self_graph
    def output_layer(self, h_, labels_):
        """Construct a softmax output layer.
        Implements:
            logits = h W + b
            loss = cross_entropy(softmax(logits), labels)
        You should define variables for the weight matrix W_out and bias term
        b_out. Initialize the weight matrix with random normal noise (use
        tf.random_normal_initializer()), and the bias term with zeros (use
        tf.zeros_initializer()).
        For the cross-entropy loss, you'll want to use
        tf.nn.sparse_softmax_cross_entropy_with_logits. This produces output of
        shape [batch_size], the loss for each example. You should use
        tf.reduce_mean to reduce this to a scalar.
        Args:
            h_: [batch_size, d] Tensor of float32, the input activations from a
                previous layer
            labels_: [batch_size] Tensor of int32, the target label ids
            num_classes: (int) the number of output classes
        Returns: (loss_, logits_)
            loss_: scalar Tensor of float32, the cross-entropy loss
            logits_: [batch_size, num_classes] Tensor of float32, the logits (hW + b)
        """
        with tf.name_scope("Output_Layer"):
            self.W_out_ = tf.get_variable("W_out", shape=[h_.get_shape()[1].value,len(self.embed[0])], initializer=tf.random_normal_initializer())
            self.b_out_ = tf.get_variable("b_out", shape=[len(self.embed[0])], initializer=tf.zeros_initializer())
            self.logits_ = tf.add(tf.matmul(h_,W_out_),b_out_) 

        with tf.name_scope("Loss"):
            self.loss_ = tf.reduce_mean(tf.square(tf.norm(labels_ - logits_)))
            self.optimizer_ = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate_)
            self.train_step_ = self.optimizer.minimize(self.loss_)
    
    @with_self_graph
    def buildModel(self, learning_rate, hidden_dims, use_dropout=True):
        with tf.name_scope("Training_Parameters"):
            self.learning_rate_ = tf.constant(learning_rate, name="learning_rate")
            self.is_training_ = tf.placeholder(tf.bool, name="is_training")
        if use_dropout:
            dropout_rate = 0.5
        else:
            dropout_rate = 0.0   

        self.a_id_ = tf.placeholder(tf.int32, [None], name="A_id")
        self.b_id_ = tf.placeholder(tf.int32, [None], name="B_id")
        self.c_id_ = tf.placeholder(tf.int32, [None], name="C_id")
        self.d_id_ = tf.placeholder(tf.int32, [None], name = "D_id")
          
        # Build embedding layer
        self.W_embed_ = self.embedding_layer()
        self.a_ = tf.nn.embedding_lookup(self.W_embed_, self.a_id_)
        self.b_ = tf.nn.embedding_lookup(self.W_embed_, self.b_id_)
        self.c_ = tf.nn.embedding_lookup(self.W_embed_, self.c_id_)
        self.labels_ = tf.nn.embedding_lookup(self.W_embed_, self.d_id_)

        # Build fully-connected layers
        self.diff_ = self.b_ - self.a_
        self.input_ = tf.concat([self.diff_, self.c_], axis=1)
        self.Deep_Layer_  = self.fully_connected_layers(self.input_, hidden_dims, activation=tf.tanh,
                                                  dropout_rate=self.dropout_rate_, is_training=self.is_training_)
        
        # Build output layer
        self.output_layer(self.Deep_Layer_, self.labels_)
        
    def trainModel(session, num_epochs, batch_size, training_file):
        train_file = open(training_file, 'r')
        train_a = []
        train_b = []
        train_c = []
        train_d = []
        for line in train_file.readlines()
            a, b, c, d = line.strip().split()
            train_a.append(self.word_to_id[a])
            train_b.append(self.word_to_id[b])
            train_c.append(self.word_to_id[c])
            train_d.append(self.word_to_id[d])
        train_file.close() 
        batches = multi_batch_generator(data, train_a, train_b, train_c, train_d)
        for (a, b, c, d) in batches:
            print(a, b, c, d)
