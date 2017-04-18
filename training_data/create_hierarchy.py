"""Create directory hierarchy for training data

Results in directory structure expected for image_dir in
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

    -- output_dir/
           |
           |-- class_0/
           |      |
           |      |-- image_0
           |      |-- image_1
           |     ...
           |
           |-- class_1/
                  |
                  |-- image_0
                  |-- image_1
                 ...

Usage:
    > python dir_1 dir_2 -o out_dir

IF GETTING PERMISSION ERRORS:
    Run with sudo:
    > sudo python dir_1 dir_2 -o out_dir
"""

import argparse
import glob
import os
import shutil
import tarfile

def get_args():
    '''Get command-line arguments to this script

    Outputs:
        in_dirs: [str]; list of directories containing training data
        out_dir: str; directory to save hierarchy
        pos_label: str; label of positive class (1)
        neg_label: str; label of negative class (0)
    '''

    class ValidatePathAction(argparse.Action):
        '''Validate directory paths'''
        def __call__(self, parser, namespace, values, option_string=None):

            if option_string:
                # Output directory may need to be created
                os.makedirs(values, exist_ok=True)
            else:
                # Check array of input directories
                for d in values:
                    if not os.path.isdir(d):
                        parser.error('Directory "{}" does not exist!'.format(d))

            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser(description='Create class hierarchy directory structure for training data')
    parser.add_argument('in_dirs', type=str, nargs='+', action=ValidatePathAction,
                        help='Directories containing training data tarballs and labels')
    parser.add_argument('-o', type=str, default='.', dest='out_dir', action=ValidatePathAction,
                        help='Directory to save new hierarchy structure')
    parser.add_argument('--pos_label', type=str, default='violence', dest='pos_label',
                        help='Label for positive class (1)')
    parser.add_argument('--neg_label', type=str, default='normal', dest='neg_label',
                        help='Label for negative class (0)')

    return parser.parse_args()

def create_class_directories(out_dir, classes):
    '''Create class directories within out_dir

    Expects list of classes in order:
        [0] => neg
        [1] => pos

    Input:
        out_dir: str; root directory for class directories
        classes: [str]; list of classes

    Output:
        class_dirs: [int] => str; dictionary mapping class label to class directory
    '''

    class_dirs = dict()
    for i, c in enumerate(classes):
        new_dir = os.path.join(out_dir, c)
        class_dirs[i] = new_dir
        os.makedirs(new_dir, exist_ok=True)

    return class_dirs

def get_tarballs(dirs):
    '''Get all tarballs in dirs

    Input:
        dirs: [str]; directories containing training data

    Output:
        tarballs: [str]; all tarballs in dirs
    '''

    tarballs = []
    for d in dirs:
        tarballs.extend(glob.glob(os.path.join(d, '*.tar.gz')))

    return tarballs

def get_frame_labels(filepath):
    '''Extract frame labels from a file

    Input:
        filepath: str; path to label file

    Output:
        labels: [int]; frame labels; 0 (neg) or 1 (pos)
    '''

    if not os.path.isfile(filepath):
        print('"{}" is not a valid file!'.format(filepath))
        return []

    with open(filepath, 'r') as f:
        labels = [int(l[:-1]) for l in f.readlines()]

    return labels

def main():
    '''Main script logic'''

    args = get_args()
    class_dirs = create_class_directories(args.out_dir, [args.neg_label, args.pos_label])
    tarballs = get_tarballs(args.in_dirs)

    for tb in tarballs:

        # Tarballs: filename.tar.gz
        # Label files: filename_labels.txt
        labels = get_frame_labels('{}_labels.txt'.format(os.path.splitext(os.path.splitext(tb)[0])[0]))

        idx = 0
        with tarfile.open(tb, 'r') as tf:
            for m in tf.getmembers():
                if m.isfile():
                    # Write only basename of file to avoid recursive directory creation
                    with open(os.path.join(class_dirs[labels[idx]], os.path.basename(m.name)), 'wb') as f:
                        shutil.copyfileobj(tf.extractfile(m), f)
                    idx += 1

if __name__ == '__main__':
    main()
