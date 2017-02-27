"""Format training input"""

import lstm.io_wrapper as iow

# TODO(gnashcraft):
# 1. Split into training, validation, testing sets
# 2. Get dimensions [batch, time, pool_values]
# 3. Get batches
# 3a. Training
# 3b. Validation
# 3c. Testing

READ_THREADS = 4

def _read_parse_example(queue):
    '''Read and parse a single sequence example

    Input:
        queue: tf queue; a queue of filenames to read

    Output:
        A single sequence example and label
    '''

    # TODO(gnashcraft): define how to parse a sequence example
    # TODO(gnashcraft): sequence class?
    context_features = {

    }
    sequence_features = {
        'activations': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'labels': tf.FixedLenSequenceFeature([], dtype=tf.int32)
    }

    reader = tf.RecordReader()
    _, example = reader.read(queue)
    context, sequence = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # TODO(gnashcrat): anything with context?

    return sequence['activations'], sequence['labels']

class Metadata():
    '''Contains metadata for data'''

    def __init__(self, n, b, t, a):
        '''Create a new Metadata

        Input:
            n: int; number of examples in the dataset
            b: int; number of batches of the dataset
            t: int; number of frames in a batch
            a: int; number of activations in an example
        '''

        self.num_examples = n
        self.batch_size = b
        self.time_steps = t
        self.activations = a

class Data():
    '''Formatted data set'''

    def __init__(self, name, filenames):
        '''Create new Data

        Input:
            name: str; name of this dataset
            filenames: [str]; list of filenames for TFRecord files
        '''

        for fn in filenames:
            if not iow.exists(fn):
                raise Exception('{} does not exist! Unable to create {} dataset.'.format(fn, name))

        self._name = name + ' Dataset'
        self._filenames = filenames
        self._use_many_readers = len(filenames) >= READ_THREADS

    def get_batch(self, batch_size, epochs=None):
        '''Read data in batches for a number of epochs

        Input:
            batch_size: int; the batch size of the data
            epochs: int; the number of epochs to read the data

        Output:
            sequences: [sequence]; a list of sequences of length batch_size
            labels: [labels]; a list of labels of length batch_size
        '''

        # NOTE: allow_smaller_final_batch causes first dimension of shape to be None
        # Therefore, operations depending on fixed batch_size will fail
        with tf.name_scope(self._name):
            queue = tf.train.string_input_producer(self._filenames, num_epochs=epochs)

            if self._use_many_readers:
                example_list = [_read_parse_example(queue) for _ in range(READ_THREADS)]
                batch_examples, batch_labels = tf.train.batch_join(
                    example_list, batch_size, dynamic_pad=True,
                    allow_smaller_final_batch=True
                )
            else:
                example, label = _read_parse_example(queue)
                batch_examples, batch_labels = tf.train.batch(
                    [example, label], batch_size, dynamic_pad=True,
                    allow_smaller_final_batch=True, num_thread=READ_THREADS
                )

            return batch_examples, batch_labels
