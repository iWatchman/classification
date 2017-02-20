"""Train an LSTM RNN from pool layer output of InceptionNet

Usage:
    For upload to Google Cloud Machine Learning
    TODO(gnashcraft)
"""

import tensorflow as tf
from utils import io

# Define optional command-line arguments
# Positional arguments are passed into argv in main()
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir',
                           '',
                           'Directory containing training data.')
tf.app.flags.DEFINE_string('labels_dir',
                           '',
                           'Directory containing training data labels.')
tf.app.flags.DEFINE_string('pool_layer_dir',
                           '',
                           'Directory containing pool layer values for training data.')
tf.app.flags.DEFINE_string('summary_dir',
                           './summaries',
                           'Directory containing summary training logs.')
tf.app.flags.DEFINE_string('output_graph',
                           './lstm_graph.pb',
                           'Filename to save trained lstm graph.')
tf.app.flags.DEFINE_float('learning_rate',
                          0.01,
                          'Initial learning rate value for training.')
tf.app.flags.DEFINE_float('decay_rate',
                          0.1,
                          'Decay rate for decreasing learning rate.')
tf.app.flags.DEFINE_integer('epochs',
                            4000,
                            'Number of iterations through training data.')
tf.app.flags.DEFINE_integer('evaluate_interval',
                            10,
                            'Epoch interval to evaluate training results.')
tf.app.flags.DEFINE_integer('val_perc',
                            10,
                            'Percent of data to use for validation set.')
tf.app.flags.DEFINE_integer('test_perc',
                            10,
                            'Percent of data to use for testing set.')
tf.app.flags.DEFINE_integer('train_secs',
                            10,
                            'Seconds of video for training batches.')
tf.app.flags.DEFINE_integer('val_secs',
                            10,
                            'Seconds of video for validation batches.')
tf.app.flags.DEFINE_integer('test_secs',
                            10,
                            'Seconds of video for testing batches.')

def main(argv=None):
    '''Main program script'''

    # TODO(gnashcraft):
    # 1. Split data into train, validation, testing sets
    # 2. Add lstm training, evaluation operations
    # 3. Run training loop
    # 3.1. Perform interval evaluations
    # 4. Perform testing evaluation
    # 5. Save trained graph
    return

if __name__ == '__main__':
    tf.app.run()
