"""Train an LSTM RNN from pool layer output of InceptionNet

Usage:
    For upload to Google Cloud Machine Learning
    TODO(gnashcraft)
"""

import input_data
import io_wrapper as iow
import json
import model
import os
import tensorflow as tf

# Define optional command-line arguments
# Positional arguments are passed into argv in main()
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir',
                           '',
                           'Directory containing training data.')
tf.app.flags.DEFINE_string('labels_dir',
                           '',
                           'Directory containing training data labels.')
tf.app.flags.DEFINE_string('save_dir',
                           '.',
                           'Root directory for saving training output.')
tf.app.flags.DEFINE_string('log_dir',
                           'logs',
                           'Directory containing model checkpoints and event logs.')
tf.app.flags.DEFINE_string('output_graph',
                           'lstm_graph.pb',
                           'Filename to save trained lstm graph.')
tf.app.flags.DEFINE_float('learning_rate',
                          0.01,
                          'Initial learning rate value for training.')
tf.app.flags.DEFINE_float('decay_rate',
                          0.8,
                          'Decay rate for decreasing learning rate.')
tf.app.flags.DEFINE_integer('decay_step',
                            100,
                            'Number of steps to increase decay rate.')
tf.app.flags.DEFINE_integer('validate_step',
                            10,
                            'Number of steps to validate training.')
tf.app.flags.DEFINE_integer('epochs',
                            5000,
                            'Number of iterations through training data.')
tf.app.flags.DEFINE_integer('val_perc',
                            10,
                            'Percent of data to use for validation set.')
tf.app.flags.DEFINE_integer('test_perc',
                            10,
                            'Percent of data to use for testing set.')
tf.app.flags.DEFINE_integer('batch_size',
                            10,
                            'Batch size for model inputs.')
tf.app.flags.DEFINE_integer('frames_per_sequence',
                            25,
                            'Number of frames in a sequence.')
tf.app.flags.DEFINE_integer('window_shift_factor',
                            2,
                            'Inverse factor of shifting across frames between sequences.')
tf.app.flags.DEFINE_integer('hidden_units',
                            256,
                            'Number of hidden units within an lstm layer.')
tf.app.flags.DEFINE_integer('num_layers',
                            1,
                            'Number of lstm layers.')
tf.app.flags.DEFINE_float('keep_prob',
                          0.8,
                          'Probability that input will retain between training layers.')
tf.app.flags.DEFINE_float('init_scale',
                          0.05,
                          'Scale for initializing variables: (-init_scale, init_scale).')

def main(argv=None):
    '''Main program script'''

    # Get hyperparameter tuning trial id
    # See https://cloud.google.com/ml/docs/how-tos/using-hyperparameter-tuning
    # for more details
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    tf_task = tf_config.get('task', {})
    tf_trial_id = tf_task.get('trial', '')

    # Create save directory
    save_dir = os.path.join(FLAGS.save_dir, tf_trial_id)
    iow.verify_dirs_exist(save_dir)

    # Collect data into sequences and split into training, validation, and testing sets
    data = input_data.Data(FLAGS.data_dir, FLAGS.labels_dir,
                           FLAGS.frames_per_sequence, FLAGS.window_shift_factor,
                           FLAGS.val_perc, FLAGS.test_perc)

    # Configure models
    train_config = valid_config = test_config =  {
        'time': data.dims[1],
        'n_act': data.dims[2],
        'hidden_units': FLAGS.hidden_units,
        'num_layers': FLAGS.num_layers
    }
    train_config['keep_prob'] = FLAGS.keep_prob
    train_config['learn_rate'] = FLAGS.learning_rate
    train_config['decay_rate'] = FLAGS.decay_rate
    train_config['decay_step'] = FLAGS.decay_step

    with tf.Graph().as_default():

        # Create models
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        train_model = model.TrainingModelFactory(model.Config(train_config), initializer)
        valid_model = model.ValidationModelFactory(model.Config(valid_config), initializer)
        test_model = model.TestingModelFactory(model.Config(test_config), intializer)

        # Add hyperparameter tuning metric summary
        # See https://cloud.google.com/ml/docs/how-tos/using-hyperparameter-tuning
        # for more details
        tf.summary.scalar('training/hptuning/metric', train_model.accuracy)

        # Using Supervisor manages checkpoints and summaries
        supervisor = tf.train.Supervisor(logdir=os.path.join(save_dir, FLAGS.log_dir))
        with supervisor.managed_session() as sess:
            for epoch in range(FLAGS.epochs):

                # Check if we need to stop
                if supervisor.should_stop():
                    break

                # Run training epoch
                acc = loss = 0.0
                count = 0
                for train_x, train_y in data.train.generate_batches(FLAGS.batch_size):
                    train_acc, train_loss = train_model.train_step(sess, train_x, train_y)
                    acc += train_acc
                    loss += train_loss
                    count += 1

                print(
                    '{} //==> [{}] => Training :: accuracy: {}, loss: {}'
                    .format(tf_trial_id, epoch + 1, acc / count, loss / count)
                )

                # Validate testing
                if epoch % FLAGS.validate_step == 0:
                    valid_x, valid_y = data.validate.get_batch(FLAGS.batch_size)
                    valid_acc, valid_loss = valid_model.check_progress(sess, valid_x, valid_y)
                    print(
                        '{} //==> [{}] => Validate :: accuracy: {}, loss: {}'
                        .format(tf_trial_id, epoch + 1, valid_acc, valid_loss)
                    )

            print('{} //==> Training complete!'.format(tf_trial_id))

            # Training complete, now do final tests
            acc = loss = 0.0
            count = 0
            for test_x, test_y in data.test.generate_batches(FLGAS.batch_size):
                test_acc, test_loss = test_model.check_progress(sess, test_x, test_y)
                acc += test_acc
                loss += test_loss
                count += 1

            print(
                '{} //==> [{}] => Testing :: accuracy: {}, loss: {}'
                .format(tf_trial_id, epoch + 1, acc / count, loss / count)
            )

            # Save output graph
            output_graph = os.path.join(save_dir, FLAGS.output_graph)
            supervisor.saver.export_meta_graph(output_graph)
            print('{} //==> Saved new graph to {}.'.format(tf_trial_id, output_graph))

if __name__ == '__main__':
    tf.app.run()
