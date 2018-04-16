
from __future__ import print_function
from __future__ import division

import time, os, shutil
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
        self.max_grad_norm_ = 1.0
        print("Reading embedding file...", end='', flush=True)
        self.readEmbedFile(embed_file)
        print("OK", flush=True)

    def readEmbedFile(self, filename):
        embed_file = open(filename, 'r')
        for line in embed_file.readlines():
            row = line.strip().split()
            self.id_to_word[len(self.vocab)] = row[0]
            self.word_to_id[row[0]] = len(self.vocab)
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
                               dropout_rate=0,is_training=False):
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
            self.logits_ = tf.add(tf.matmul(h_,self.W_out_),self.b_out_) 

        with tf.name_scope("Loss"):
            self.loss_ = tf.reduce_mean(tf.norm(labels_ - self.logits_))
            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.learning_rate_)
            gradients_, variables_ = zip(*self.optimizer_.compute_gradients(self.loss_))
            clipped_grads_, _ = tf.clip_by_global_norm(gradients_, self.max_grad_norm_)
            self.train_step_ = self.optimizer_.apply_gradients(zip(clipped_grads_,variables_))
    
    @with_self_graph
    def buildModel(self, learning_rate, hidden_dims, use_dropout=True):
        print("Building model graph...", end='', flush=True)
        with tf.name_scope("Training_Parameters"):
            self.learning_rate_ = tf.constant(learning_rate, name="learning_rate")
            self.is_training_ = tf.placeholder(tf.bool, name="is_training")
        if use_dropout:
            self.dropout_rate_ = 0.5
        else:
            self.dropout_rate_ = 0.0   

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
        
        print("OK", flush=True)

    @with_self_graph    
    def trainModel(self, num_epochs, batch_size, training_file, savedir):
        print("Training model! Please wait...\n", flush=True)
        checkpoint_filename = os.path.join(savedir, "checkpoint")
        trained_filename = os.path.join(savedir, "trained_model")
        train_file = open(training_file, 'r')
        train_a = []
        train_b = []
        train_c = []
        train_d = []
        for line in train_file.readlines():
            if line[0] != ":" and len(line.strip().split()) == 4:
                a, b, c, d = line.strip().split()
                if a in self.vocab and b in self.vocab and c in self.vocab and d in self.vocab:
                    train_a.append(self.word_to_id[a])
                    train_b.append(self.word_to_id[b])
                    train_c.append(self.word_to_id[c])
                    train_d.append(self.word_to_id[d])
        train_file.close() 
        batches = multi_batch_generator(batch_size, train_a, train_b, train_c, train_d)
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # Clear old log directory
        shutil.rmtree(savedir, ignore_errors=True)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(42)

            session.run(initializer)

            for epoch in range(1,num_epochs+1):
                batches = multi_batch_generator(batch_size, train_a, train_b, train_c, train_d)
                print("[epoch {:d}] Starting epoch {:d}".format(epoch, epoch), flush=True)
                # Run a training epoch.
                start_time = time.time()
                total_cost = 0.0
                total_batches = 0.0
                for i, (a_id, b_id, c_id, d_id) in enumerate(batches):
                    feed_dict = {self.is_training_: True,
                                 self.a_id_: a_id,
                                 self.b_id_: b_id,
                                 self.c_id_: c_id,
                                 self.d_id_: d_id}
                    cost, train_step = session.run([self.loss_, self.train_step_], feed_dict)
                    total_cost += cost
                    total_batches = i+1
                total_time = time.time() - start_time
                avg_cost = total_cost/total_batches
                print("[epoch {:d}] Completed in {:.3f}, loss: {:f}".format(epoch, total_time,avg_cost), flush=True)
    
                # Save a checkpoint
                saver.save(session, checkpoint_filename, global_step=epoch)
            print("\nTraining finished. Persisting model... ", end='', flush=True)
            # Save final model
            saver.save(session, trained_filename)
            print("OK", flush=True)
    
    @with_self_graph
    def predict(self, input_file, savedir):
        trained_filename = os.path.join(savedir, "trained_model")
        infile = open(input_file, 'r')
        a = []
        b = []
        c = []
        for i, line in enumerate(infile.readlines()):
            words = line.strip().split()
            if len(words) >= 3:
                if words[0] in self.vocab and words[1] in self.vocab and words[2] in self.vocab:
                    a.append(self.word_to_id[words[0]])
                    b.append(self.word_to_id[words[1]])
                    c.append(self.word_to_id[words[2]])
                else:
                    print("Line %d has Out-Of-Vocabulary words, skipping..." % i)
            else:
                print("Line %d has fewer words than expected, skipping..." % i)
        infile.close()
        saver=tf.train.Saver()
        with tf.Session(graph=self.graph) as session:
            saver.restore(session, trained_filename)
            feed_dict = {self.is_training_: False,
                         self.a_id_: a,
                         self.b_id_: b,
                         self.c_id_: c}
            logits = session.run([self.logits_], feed_dict)
        return logits 
