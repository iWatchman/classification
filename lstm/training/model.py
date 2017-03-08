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

def _accuracy_ops(targets, predictions):
    '''Calculate aggregated accuracy

    Input:
        targets: tf.Tensor [batch_size, time_steps]; the target labels
        predictions: tf.Tensor [batch_size, time_steps, classes]; confidences per class

    Output:
        accuracy: the accuracy operation
        updates: update operations for aggregating across batches
        resets: reset operations to reset update counters following an epoch
    '''

    # Initial counter values (this is only called once)
    incorrect_counter = tf.Variable(0, trainable=False)
    correct_counter = tf.Variable(0, trainable=False)

    # Correct and incorrect counts for a batch
    missed = tf.not_equal(targets, tf.cast(tf.arg_max(predictions, 2), tf.int32))
    incorrect = tf.reduce_sum(tf.cast(missed, tf.int32))
    correct = tf.reduce_sum(tf.cast(tf.logical_not(missed), tf.int32))

    # Keep running counts across all batches
    update_incorrect_counter = tf.assign_add(incorrect_counter, incorrect)
    update_correct_counter = tf.assign_add(correct_counter, correct)

    # Reset counts at the end of every epoch
    reset_incorrect_counter = tf.assign(incorrect_counter, 0)
    reset_correct_counter = tf.assign(correct_counter, 0)

    # Accuracy operation
    accuracy = tf.div(tf.cast(correct_counter, tf.float32), tf.cast(correct_counter + incorrect_counter, tf.float32))

    return accuracy, [update_incorrect_counter, update_correct_counter], [reset_incorrect_counter, reset_correct_counter]

def _cost_ops(loss):
    '''Calculate aggregate cost

    Input:
        loss: tf.Tensor; the batch loss

    Output:
        cost: the cost operation
        updates: update operations for aggregating across batches
        resets: reset operations to reset update counters following and epoch
    '''

    # Initial counter values (this is only called once)
    total = tf.Variable(0.0, trainable=False)
    count = tf.Variable(0, trainable=False)

    # Keep running counters across all batches
    update_total = tf.assign_add(total, loss)
    update_count = tf.assign_add(count, 1)

    # Reset counters at the end of every epoch
    reset_total = tf.assign(total, 0.0)
    reset_count = tf.assign(count, 0)

    # Cost operation
    cost = tf.div(total, tf.cast(count, tf.float32))

    return cost, [update_total, update_count], [reset_total, reset_count]

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
        self._inputs = tf.placeholder(tf.float32, [None, config.time, config.n_act], name='inputs')
        self._labels = tf.placeholder(tf.int32, [None, config.time], name='labels')

        # Trainable variables for linear activation layer
        weights = tf.get_variable('weights', [config.hidden_units, config.classes], tf.float32)
        bias = tf.get_variable('bias', [config.classes], tf.float32)

        # Graph
        if config.keep_prob < 1.0:
            inputs = tf.nn.dropout(self._inputs, config.keep_prob)
        cell = _lstm_cell(config.hidden_units, config.keep_prob, config.num_layers)
        outputs, states = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, config.hidden_units])
        logits = tf.reshape(tf.nn.xw_plus_b(outputs, weights, bias), [-1, config.time, config.classes], name='logits')

        # Batch predictions
        self._preds = tf.nn.softmax(logits, name='predictions')

        # Calculate loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._labels, name='cross_entropy')
        self._loss = tf.reduce_sum(cross_entropy, name='loss')

        # Calculate aggregate accuracy and cost
        self._accuracy, accuracy_updates, accuracy_resets = _accuracy_ops(self._labels, self._preds)
        self._cost, cost_updates, cost_resets = _cost_ops(self._loss)
        self._updates = accuracy_updates + cost_updates
        self._resets = accuracy_resets + cost_resets

        # Add summaries later if desired with add_summaries
        self._summary_op = None

    def add_summaries(self, summaries):
        '''Add summaries to the Model

        Input:
            summaries: [string tensor]; summaries to add
        '''

        if self._summary_op is not None:
            summaries.append(self._summary_op)

        self._summary_op = tf.summary.merge(summaries)

    def check(self, sess, inputs=None, labels=None, generator=None, updates=[]):
        '''Check the accuracy and cost of the current model

        May be used either for a single batch or an entire epoch.
        For single batch use, must supply inputs and labels.
        For entire epoch use, must supply a generator.

        Input:
            sess: tf.Session; current session
            inputs: tensor [batch, time, activations]; batch sequential outputs from inception net
            labels: tensor [batch, time]; labels for each activation input
            generator: generates batch inputs and labels
            updates: [tf.Tensor]; list of tensors to run updates on per batch

        Output:
            summary: serialized protobuf; computed model summaries
            accuracy: float; the accuracy of the model at predicting inputs
            cost: float; the cost of the model in predicting inputs
        '''

        fetches = {
            'summary': self._summary_op,
            'acc': self._accuracy,
            'cost': self._cost
        }

        if generator is not None:
            for inputs, labels in generator:
                feed = {self._inputs: inputs, self._labels: labels}
                sess.run(self._updates + updates, feed_dict=feed)
        else:
            feed = {self._inputs: inputs, self._labels: labels}
            sess.run(self._updates + updates, feed_dict=feed)

        output = sess.run(fetches)
        sess.run(self._resets)

        return output['summary'], output['acc'], output['cost']

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
            config: see Model class
        '''

        super(TrainModel, self).__init__(config)

        # Learning rate will decay over time
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        self._increment_gs = tf.assign_add(self._global_step, 1)
        self._lr = tf.train.exponential_decay(config.learn_rate,
                                              self._global_step,
                                              config.decay_step,
                                              config.decay_rate,
                                              staircase=True,
                                              name='learn_rate')

        # Training operation
        self._train_op = (
            tf.train.GradientDescentOptimizer(self._lr)
            .minimize(self._loss)
        )

    def train_step(self, sess, generator):
        '''Perform training for an entire epoch

        Input:
            See Model.check()

        Output:
            see Model.check()
        '''

        output = self.check(sess, generator=generator, updates=[self._train_op])
        sess.run(self._increment_gs)
        return output

    @property
    def global_step(self):
        return self._global_step

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
        The created TrainModel
    '''

    with tf.name_scope('Training'):
        with tf.variable_scope('Model', reuse=None, initializer=init):
            model = TrainModel(config)
        model.add_summaries([
            tf.summary.scalar('Accuracy', model.accuracy),
            tf.summary.scalar('Loss', model.cost),
            tf.summary.scalar('Learning_Rate', model.learn_rate)
        ])

    return model

def ValidationModelFactory(config, init, hptuning=False):
    '''Create a new Model for validation

    Wraps model and summary writers in appropriate variable and name scopes.

    Input:
        config: see Model class;
        init: Initializer for variables
        hptuning: bool; True if tuning hyperparameters, False otherwise

    Output:
        The created Model
    '''

    with tf.name_scope('Validation'):
        with tf.variable_scope('Model', reuse=True, initializer=init):
            model = Model(config)
        model.add_summaries([
            tf.summary.scalar('Accuracy', model.accuracy),
            tf.summary.scalar('Loss', model.cost)
        ])

    # Add hyperparameter tuning summary (if applicable)
    # See https://cloud.google.com/ml/docs/how-tos/using-hyperparameter-tuning
    # for more details
    if hptuning:
        model.add_summaries([tf.summary.scalar('training/hptuning/metric', model.accuracy)])

    return model

def TestingModelFactory(config, init):
    '''Create a new Model for testing

    Wraps model in appropriate variable and name scopes.

    Input:
        config: see Model class;
        init: Initializer for variables

    Output:
        The created Model
    '''

    with tf.name_scope('Testing'):
        with tf.variable_scope('Model', reuse=True, initializer=init):
            model = Model(config)
        model.add_summaries([
            tf.summary.scalar('Accuracy', model.accuracy),
            tf.summary.scalar('Loss', model.cost)
        ])

    return model
