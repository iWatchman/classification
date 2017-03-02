"""Format training input"""

import io_wrapper as iow
import multiprocessing as mp
import numpy as np
import os
import re

THREAD_STOP = 'STOP'

def _get_video_name(filename):
    '''Get the video name from a label or frame filename

    Assumes filename follows one of the convetions:
        `videoName_labels.txt`
        'videoName_frameNumber.jpg.txt'

    Input:
        filename: str; the name of the file

    Output:
        The name of the corresponding video
    '''

    match = re.search(r"(.*)_", filename)
    return match.groups()[0]

def _get_frame_number(filename):
    '''Get the frame number from the file name of a frame

    Assumes filename is of the form `videoName_frameNumber.jpg.txt`.

    Input:
        filename: str; name of frame file

    Output:
        int; The frame number of the frame
    '''

    # Assumes filename is of the form `videoName_frameNumber.jpg.txt`
    match = re.search(r"_([0-9]*)\.jpg\.txt", filename)
    n = match.groups()[0]

    # Frames are numbered beginning at 0001 and we need to convert to 0-index
    return int(n) - 1

def _read_file(filename, cast=float, delim=','):
    '''Read a file containing values separated by a delimiter

    Input:
        filename: str; the name of the file to read
        cast: type; a type to cast the vales. defaults to float
        delim: str; the delimiter of the data. defaults to `,`

    Output:
        A list of values read
    '''

    with tf.gfile.FastGFile(filename, 'r') as f:
        f_str = f.read()

    # Cast each value, skipping any empty strings
    return [cast(x) for x in f_str.split(delim) if x]

def _parser(in_q, act_q, lbl_q):
    '''Parser worker process

    Input:
        in_q: multiprocessing.Queue; work queue
        act_q: multiprocessing.Queue; sequence activations queue
        lbl_q: multiprocessing.Queue; sequence labels queue
    '''

    for seq in iter(in_q.get, THREAD_STOP):
        activations, labels = seq.parse()
        act_q.put(activations)
        lbl_q.put(labels)

def _parse_batch(sequences, threads):
    '''Get a batch of parsed sequences

    Uses mulitprocessing to parallelize sequence parsing.

    Input:
        sequences: np.array([Sequence]); numpy array of Sequences
        threads: int; number of threads to use in parallel

    Output:
        Parsed batch of sequence activations and labels
    '''

    seq_queue = mp.Queue()
    act_queue = mp.Queue()
    lbl_queue = mp.Queue()

    for seq in sequences:
        seq_queue.put(seq)

    for _ in range(threads):
        mp.Process(target=_parser, args=(seq_queue, act_queue, lbl_queue)).start()

    for _ in range(threads):
        seq_queue.put(THREAD_STOP)

    batch_acts = []
    batch_lbls = []
    for _ in range(len(sequences)):
        batch_acts.append(act_queue.get())
        batch_lbls.append(lbl_queue.get())

    return np.array(batch_acts), np.array(batch_lbls)

def _create_video_sequences(data_dir, label_file, time_steps, shift):
    '''Create Sequences for a video

    Input:
        data_dir: str; directory containing frame files
        label_file: str; a label file for a video
        time_steps: int; number of frames in a Sequence
        shift: int; number of frames to shift for sliding window

    Output:
        A list of Sequences
    '''

    # Find all frame files for the video
    # Assumes frame filenames are of the form `videoName_frameNumber.jpg.txt`
    video_name = _get_video_name(label_file)
    frame_files = tf.gfile.Glob(os.path.join(data_dir, '*', video_name + '_*.jpg.txt'))
    frame_files = sorted(frame_files, key=lambda x: _get_frame_number(x))

    cur = 0
    done = False
    sequences = []
    while not done and cur < len(frame_files):

        # Push cur back so that last sequence has the same number of frames
        # as all other sequences
        if cur + time_steps >= len(frame_files):
            cur = len(frame_files) - time_steps
            done = True

        sequences.append(Sequence(frame_files[cur:cur + time_steps], label_file))

        cur += shift

    return sequences

def _video_sequencer(in_q, out_q):
    '''Video sequencer worker process

    Input:
        in_q: multiprocessing.Queue; work queue
        out_q; multiprocessing.Queue; sequence array queue
    '''

    for args in iter(in_q.get, THREAD_STOP):
        out_q.put(_create_video_sequences(**args))

def _create_sequences(data_dir, label_dir, time_steps, shift, threads):
    '''Create a list of sequences from the data

    Input:
        data_dir: str; directory containing frame data
        label_dir: str; directory containing video labels
        time_steps: int; number of frames in a sequence
        shift: int; shift of sliding window
        threads: int; number of threads to use in parallel

    Output:
        A list of Sequences
    '''

    arg_queue = mp.Queue()
    seq_queue = mp.Queue()

    # Assume all labels follow the convention 'videoName_labels.txt'
    labels = tf.gfile.Glob(os.path.join(label_dir, '*_labels.txt'))

    for label in labels:
        args = {
            'data_dir': data_dir,
            'label_file': label,
            'time_steps': time_steps,
            'shift': shift
        }
        arg_queue.put(args)

    for _ in range(threads):
        mp.Process(target=_video_sequencer, args=(arg_queue, seq_queue)).start()

    for _ in range(threads):
        arg_queue.put(THREAD_STOP)

    sequences = []
    for _ in range(len(labels)):
        sequences.extend(seq_queue.get())

    return sequences

class Sequence():
    '''A sequence of training data'''

    def __init__(self, frames, label_file):
        '''Create a new Sequence

        Assumes frames are in order of the sequence and that each frame
        file name is of the form `videoName_frameNumber.jpg.txt`.

        label_file should follow naming convention: `videoName_labels.txt`.

        Input:
            frames: [str]; list of file names of frames in the sequence
            label_file: str; file name of labels file
        '''

        # Get starting and ending indices of the sequence
        self._range = {
            'start': _get_frame_number(frames[0]),
            'end': _get_frame_number(frames[-1])
        }

        self._frames = frames
        self._labels = label_file

    def parse(self):
        '''Parse frame activations and labels in the sequence

        Output:
            activations: [[float]]; the activations of each frame
            labels: [int]; the labels of each frame
        '''

        activations = [_read_file(frame) for frame in self._frames]
        labels = _read_file(self._labels, cast=int, delim='\n')

        # Only pull the labels associated with this sequence
        labels = labels[self._range['start']:self.range['end'] + 1]

        return activations, labels

    @property
    def num_activations(self):
        return len(_read_file(self._frames[0]))

    @property
    def time_steps(self):
        return len(self._frames)

class Dataset():
    '''Formatted dataset of Sequences'''

    def __init__(self, sequences, threads=4):
        '''Create a new Dataset

        Input:
            sequences: [Sequence]; a list of Sequences
            threads: int; number of threads to generate batches in parallel
        '''

        self._threads = threads
        self._seqs = np.array(sequences)
        self._idxs = np.random.permutation(len(sequences))
        self._cur = 0
        self._epochs = 0

    def get_batch(self, batch_size):
        '''Get a single batch of sequences

        Input:
            batch_size: int; size of the batch

        Output:
            A single batch of sequence activation and labels
        '''

        end = self._cur + batch_size
        indices = self._idxs[cur:end]
        self._cur = end
        batch_seqs = self._seqs[indices]

        # Reset at the end of an epoch
        if self._cur >= len(self._seqs):
            self._idxs = np.random.permutation(len(sequences))
            self._cur = 0
            self._epochs += 1

        return _parse_batch(batch_seqs, self._threads)

    def generate_batches(self, batch_size):
        '''Yield batch sequences for a full epoch

        Input:
            batch_size: int; size of the batch

        Yields:
            Parsed sequences and corresponding labels in batches
        '''

        cur = 0
        idxs = np.random.permutation(len(self._seqs))
        self._epochs += 1

        while cur < len(self._seqs):

            end = cur + batch_size
            indices = idxs[cur:end]
            cur = end
            batch_seqs = self._seqs[indices]

            yield _parse_batch(batch_seqs, self._threads)

    @property
    def epochs(self):
        return self._epochs

class Data():
    '''Training, validation, and testing datasets'''

    def __init__(self, data_dir, lbls_dir, time_steps, shift_factor, val_perc, test_perc, threads=4):
        '''Create a new Data

        Labels are assumed to be one label file per video. Data are considered
        one file per frame. Each frame label is one line of the label file.

        Splits all data in directory at data_dir into training, validation,
        and testing Datasets. Validation and testing Datasets are split
        based on percentages in val_perc and test_perc respectfully.

        Input:
            data_dir: str; directory containing data
            lbls_dir: str; directory containing data labels
            time_steps: int; number of frames in a sequence
            shift_factor: int inverse factor of time_steps to determine shift of sliding window
            val_perc: int; percentage of all data to use for validation
            test_perc: int; percentage of all data to use for testing
            threads: int; number of threads to generate batches in parallel
        '''

        if not iow.exists(data_dir):
            raise Exception('Unable to read data from {}. Directory does not exist!'.format(data_dir))

        if not iow.exists(lbls_dir):
            raise Exception('Unable to read labels from {}. Directory does not exist!'.format(lbls_dir))

        sequences = _create_sequences(data_dir, lbls_dir, time_steps, int(time_steps / shift_factor), threads)
        self._dims = [-1, time_steps, sequences[0].num_activations]

        # Shuffle and split Sequences into Datasets
        test_idx = int((len(sequences) * test_perc) / 100)
        val_idx = int((len(sequences) * val_perc) / 100) + test_idx
        np.random.shuffle(sequences)

        self._test = Dataset(sequences[:test_idx])
        self._validate = Dataset(sequences[test_idx:val_idx])
        self._train = Dataset(sequences[val_idx:])

    @property
    def dims(self):
        return self._dims

    @property
    def test(self):
        return self._test

    @property
    def validate(self):
        return self._validate

    @property
    def train(self):
        return self._train





################################################################################
#
#                       TFRecords implementation code
#
################################################################################
#
# READ_THREADS = 4
#
# def _read_parse_example(queue):
#     '''Read and parse a single sequence example
#
#     Input:
#         queue: tf queue; a queue of filenames to read
#
#     Output:
#         A single sequence example and label
#     '''
#
#     # TODO(gnashcraft): define how to parse a sequence example
#     # TODO(gnashcraft): sequence class?
#     context_features = {
#
#     }
#     sequence_features = {
#         'activations': tf.FixedLenSequenceFeature([], dtype=tf.float32),
#         'labels': tf.FixedLenSequenceFeature([], dtype=tf.int32)
#     }
#
#     reader = tf.RecordReader()
#     _, example = reader.read(queue)
#     context, sequence = tf.parse_single_sequence_example(
#         serialized=example,
#         context_features=context_features,
#         sequence_features=sequence_features
#     )
#
#     # TODO(gnashcrat): anything with context?
#
#     return sequence['activations'], sequence['labels']
#
# class Metadata():
#     '''Contains metadata for data'''
#
#     def __init__(self, n, b, t, a):
#         '''Create a new Metadata
#
#         Input:
#             n: int; number of examples in the dataset
#             b: int; number of batches of the dataset
#             t: int; number of frames in a batch
#             a: int; number of activations in an example
#         '''
#
#         self.num_examples = n
#         self.batch_size = b
#         self.time_steps = t
#         self.activations = a
#
# class Data():
#     '''Formatted data set'''
#
#     def __init__(self, name, filenames):
#         '''Create new Data
#
#         Input:
#             name: str; name of this dataset
#             filenames: [str]; list of filenames for TFRecord files
#         '''
#
#         for fn in filenames:
#             if not iow.exists(fn):
#                 raise Exception('{} does not exist! Unable to create {} dataset.'.format(fn, name))
#
#         self._name = name + ' Dataset'
#         self._filenames = filenames
#         self._use_many_readers = len(filenames) >= READ_THREADS
#
#     def get_batch(self, batch_size, epochs=None):
#         '''Read data in batches for a number of epochs
#
#         Input:
#             batch_size: int; the batch size of the data
#             epochs: int; the number of epochs to read the data
#
#         Output:
#             sequences: [sequence]; a list of sequences of length batch_size
#             labels: [labels]; a list of labels of length batch_size
#         '''
#
#         # NOTE: allow_smaller_final_batch causes first dimension of shape to be None
#         # Therefore, operations depending on fixed batch_size will fail
#         with tf.name_scope(self._name):
#             queue = tf.train.string_input_producer(self._filenames, num_epochs=epochs)
#
#             if self._use_many_readers:
#                 example_list = [_read_parse_example(queue) for _ in range(READ_THREADS)]
#                 batch_examples, batch_labels = tf.train.batch_join(
#                     example_list, batch_size, dynamic_pad=True,
#                     allow_smaller_final_batch=True
#                 )
#             else:
#                 example, label = _read_parse_example(queue)
#                 batch_examples, batch_labels = tf.train.batch(
#                     [example, label], batch_size, dynamic_pad=True,
#                     allow_smaller_final_batch=True, num_thread=READ_THREADS
#                 )
#
#             return batch_examples, batch_labels
