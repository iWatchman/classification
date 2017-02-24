"""Create LSTM networks"""

import tensorflow as tf

# TODO(gnashcraft):
# 1. __init__()

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
        return tf.contrib.rnn.BasicLSTMCell(hidden_units)

    def dropout_cell():
        return tf.contrib.rnn.DropoutWrapper(cell(), output_keep_prob=keep_prob)

    cell_layer = dropout_cell if keep_prob < 1.0 else cell
    return tf.contrib.rnn.MultiRNNCell([cell_layer() for _ in range(num_layers)])

class Model():
    '''LSTM Network'''

    def __init__(self, config):
        '''Create a new Model

        Input:
            config: Config; configuration parameters
        '''

        # Model inputs
        # Size: [batches, time, pool_values]
        # TODO(gnashcraft): get input sizes from input config
        inputs = tf.placeholder(tf.float32, [None, None, None], name='inputs')
        labels = tf.placeholder(tf.float32, [None, None], name='labels')

        # Trainable variables for linear activation layer
        # TODO(gnashcraft): should probably get number of label options instead of hardcode
        weights = tf.get_variable('weights', [config.hidden_units, 2], tf.float32)
        bias = tf.get_variable('bias', [2], tf.float32)

        # Graph
        cell = _lstm_cell(config.hidden_units, config.keep_prob, config.num_layers)
        # TODO(gnashcraft): get batch size from input config
        initial_state = cell.zero_state(None, tf.float32)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        # TODO(gnashcraft): Want logits.size = [batch, time, 2]
        logits = tf.nn.xw_plus_b(outputs, weights, bias, name='logits')

        # Output calculations
        # TODO(gnashcraft):
        # 1. cost: logits - [batch, time, 2] and labels - [batch, time] and name
        # 2. accuracy: this and name
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self._accuracy = 1.0

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

class TrainModel(Model):
    '''LSTM Network for training'''

    def __init__(self, config):
        '''Create a new TrainModel

        Input:
            config: see Model class;
        '''

        super(TrainModel, self).__init__(config)

        # Learning rate can be updated
        self._lr = tf.Variable(config.learn_rate, trainable=False, name='lr')
        self._new_lr = tf.placeholder(tf.float32, [], name='new_lr')
        self._update_lr = tf.assign(self._lr, self._new_lr, name='update_lr')

        # Training operation
        self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._cost)

    def update_learnrate(self, sess, lr):
        '''Update model learning rate for training

        Input:
            sess: tf.Session; current session
            lr: float; the new learning rate
        '''

        sess.run(self._update_lr, feed_dict={self._new_lr: lr})

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
