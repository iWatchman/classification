"""Create LSTM networks"""

import tensorflow as tf

def _lstm_cell(hidden_units, keep_prob, num_layers):
    '''Create multi-layered LSTM Cell

    Input:
        hidden_units: int; number of hidden units in each lstm layer
        keep_prob: float; keep probability for dropout
        num_layers: int; number of lstm layers

    Output:
        A multi-layered lstm cell
    '''

    def cell():
        # This is r1.0
        #return tf.contrib.rnn.BasicLSTMCell(hidden_units)
        # This is r0.11
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_units)

    def dropout_cell():
        # This is r1.0
        #return tf.contrib.rnn.DropoutWrapper(cell(), output_keep_prob=keep_prob)
        # This is r0.11
        return tf.nn.rnn_cell.DropoutWrapper(cell(), output_keep_prob=keep_prob)

    cell_layer = dropout_cell if keep_prob < 1.0 else cell
    # This is r1.0
    #return tf.contrib.rnn.MultiRNNCell([cell_layer() for _ in range(num_layers)])
    # This is r0.11
    return tf.nn.rnn_cell.MultiRNNCell([cell_layer() for _ in range(num_layers)])

class Config(object):
    '''Model configuration'''

    def __init__(self, kwargs):
        '''Create a new Config

        Input:
            time: [required] - int; number of frames in a sequence
            n_act: [required] - int; number of activations in a frame
            batch_size: [optional] - int; number of sequences in a batch
            classes: [optional] - int; number of target classes
            hidden_units: [optional] - int; number of hidden units in lstm layer
            num_layers: [optional] - int; number of lstm layers
            learn_rate: [optional] - float; initial learning rate for training
            decay_rate: [optional] - float; rate of decay for learning rate
            decay_step: [optional] - int; number of steps to wait before decaying learning rate
            keep_prob: [optional] - float; probability of keeping inputs
        '''

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Defaults
        if not hasattr(self, 'batch_size'):
            self.batch_size = 10
        if not hasattr(self, 'classes'):
            self.classes = 2
        if not hasattr(self, 'hidden_units'):
            self.hidden_units = 256
        if not hasattr(self, 'num_layers'):
            self.num_layers = 1
        if not hasattr(self, 'learn_rate'):
            self.learn_rate = 0.1
        if not hasattr(self, 'decay_rate'):
            self.decay_rate = 0.8
        if not hasattr(self, 'decay_step'):
            self.decay_step = 20
        if not hasattr(self, 'keep_prob'):
            self.keep_prob = 1.0

class Model(object):
    '''LSTM Network'''

    def __init__(self, config):
        '''Create a new Model

        Input:
            config: Config; configuration parameters
        '''

        # Model inputs
        # Size: [batches, time, pool_values]
        self._inputs = inputs = tf.placeholder(tf.float32, [None, config.time, config.n_act], name='inputs')
        self._labels = labels = tf.placeholder(tf.int64, [None, config.time], name='labels')

        # Trainable variables for linear activation layer
        weights = tf.get_variable('weights', [config.hidden_units, config.classes], tf.float32)
        bias = tf.get_variable('bias', [config.classes], tf.float32)

        # Graph
        if config.keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        cell = _lstm_cell(config.hidden_units, config.keep_prob, config.num_layers)
        #initial_state = cell.zero_state([config.batch_size], tf.float32)
        #outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, config.hidden_units])
        logits = tf.nn.xw_plus_b(outputs, weights, bias, name='logits')

        # Batch predictions
        self._preds = preds = tf.reshape(tf.nn.softmax(logits), [-1, config.time, config.classes], name='predictions')

        # Calculate accuracy
        missed = tf.not_equal(labels, tf.arg_max(preds, 2), name='missed')
        self._accuracy = tf.reduce_mean(tf.cast(missed, tf.float32), name='accuracy')

        # Calculate cost
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
        self._cost = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1]), name='cost')

    def check_progress(self, sess, inputs, labels):
        '''Check the accuracy and cost of the current model

        Inputs:
            sess: tf.Session; current session
            inputs: tensor [batch, time, activations]; batch sequential outputs from inception net
            labels: tensor [batch, time]; labels for each activation input

        Outputs:
            accuracy: float; The accuracy of the model at predicting inputs
            cost: float; The cost of the model in predicting inputs
        '''

        fetches = {
            'acc': self._accuracy,
            'cost': self._cost
        }

        feed = {
            self._inputs: inputs,
            self._labels: labels
        }

        output = sess.run(fetches, feed_dict=feed)
        return output['acc'], output['cost']

    def predict(self, sess, inputs):
        '''Predict a batch of pool_layer activations

        Input:
            sess: tf.Session; current session
            inputs: tensor [batch, time, activations]; batch sequential outputs from inception net

        Output:
            A tensor [batch, time, num_classes] of predictions of each class for each timestep in each batch.
        '''

        return sess.run(self._preds, feed_dict={self._inputs: inputs})

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def cost(self):
        return self._cost

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def predictions(self):
        return self._preds

class TrainModel(Model):
    '''LSTM Network for training'''

    def __init__(self, config):
        '''Create a new TrainModel

        Input:
            config: see Model class;
        '''

        super(TrainModel, self).__init__(config)

        # Learning rate will decay over time
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self._lr = tf.train.exponential_decay(config.learn_rate,
                                              global_step,
                                              config.decay_step,
                                              config.decay_rate,
                                              name='learn_rate')

        # Training operation
        self._train_op = (
            tf.train.GradientDescentOptimizer(self._lr)
            .minimize(self._cost, global_step=global_step)
        )

    def train_step(self, sess, inputs, labels):
        '''Perform a training step

        Input:
            sess: tf.Session; current session
            inputs: tensor [batch, time, activations]; batch sequential outputs from inception net
            labels: tensor [batch, time]; labels for each activation input

        Output:
            accuracy: float; The accuracy of the model at predicting inputs
            cost: float; The cost of the model in predicting inputs
        '''

        fetches = {
            'train': self._train_op,
            'acc': self._accuracy,
            'cost': self._cost
        }

        feed = {
            self._inputs: inputs,
            self._labels: labels
        }

        output = sess.run(fetches, feed_dict=feed)
        return output['acc'], output['cost']

    @property
    def learn_rate(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def TrainingModelFactory(config, init):
    '''Create a new TrainModel for training

    Wraps model and summary writers in appropriate variable and name scopes.

    Input:
        config: see Model class;
        init: Initializer for variables

    Output:
        A new TrainModel
    '''

    with tf.name_scope('Training'):
        with tf.variable_scope('Model', reuse=None, initializer=init):
            model = TrainModel(config)
        tf.summary.scalar('Accuracy', model.accuracy)
        tf.summary.scalar('Loss', model.cost)
        tf.summary.scalar('Learning Rate', model.learn_rate)

    return model

def ValidationModelFactory(config, init):
    '''Create a new Model for validation

    Wraps model and summary writers in appropriate variable and name scopes.

    Input:
        config: see Model class;
        init: Initializer for variables

    Output:
        A new Model for validation
    '''

    with tf.name_scope('Validation'):
        with tf.variable_scope('Model', reuse=True, initializer=init):
            model = Model(config)
        tf.summary.scalar('Accuracy', model.accuracy)
        tf.summary.scalar('Loss', model.cost)

    return model

def TestingModelFactory(config, init):
    '''Create a new Model for testing

    Wraps model in appropriate variable and name scopes.

    Input:
        config: see Model class;
        init: Initializer for variables

    Output:
        A new Model for testing
    '''

    with tf.name_scope('Testing'):
        with tf.variable_scope('Model', reuse=True, initializer=init):
            model = Model(config)

    return model
