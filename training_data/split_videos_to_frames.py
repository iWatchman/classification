"""Split videos into frames

NOTE: Uses ffmpeg (https://ffmpeg.org/) for converting video to frames.

Example Usage:
  > python split_videos_to_frames video_dir mp4 -o output_dir

  Finds all videos of mp4 format in video_dir, splits each video into frames,
  and saves each collection of frames as a zipped tarball. Each tarball has the
  same basename as the corresponding video. All tarballs are saved in output_dir.

Help:
  > python split_videos_to_frames -h
"""

def get_args():
    '''Get command-line arguments to this script.

    Output:
        input: string; directory containing video files
        output: string; directory to save image frames
        verbose: bool; print verbose information (True) or not (False)
    '''

    import argparse

    parser = argparse.ArgumentParser(description='Split video files into image frames.')
    parser.add_argument('input', type=str,
                        help='Directory containing video files')
    parser.add_argument('ftype', type=str, choices=['mp4'],
                        help='Video file format of input video files')
    parser.add_argument('-o', type=str, dest='output', default='',
                        help='Output directory for image frames')
    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='Print verbose output')

    return parser.parse_args()

def main(in_dir, out_dir, ftype, verbose):
    '''Main script logic

    Input:
        in_dir: string; directory containing video files
        out_dir: string; directory to save image frames
        ftype: string; file type for input video files
        verbose: bool; print verbose information (True) or not (False)
    '''

    import glob
    from os import path
    import subprocess as sp
    import tarfile
    import tempfile

    frame_counts = dict()

    # Grab all video files
    if verbose:
        print('Searching for {} files in {}'.format(ftype, in_dir))
    files = glob.glob(path.join(in_dir, '*.' + ftype))
    if verbose:
        print('Found {} videos'.format(len(files)))

    for f in files:
        fname = os.path.splitext(os.path.basename(f))[0]
        if verbose:
            print('Splitting {} into frames'.format(fname))

        with tempfile.TemporaryDirectory() as tmpdir:
            if verbose:
                print('Creating temporary directory at {}'.format(tmpdir))

            # Run ffmpeg to split video into frames
            # NOTE: Need to have ffmpeg installed
            ffmpeg = [  'ffmpeg',
                        '-i', f,
                        os.path.join(tmpdir, '{}_%4d.jpg'.format(fname))]
            sp.run(ffmpeg)

            # Save number of frames per filename
            frames = glob.glob(path.join(tmpdir, '*.jpg'))
            frame_counts[fname] = len(frames)
            if verbose:
                print('Found {} frames'.format(len(frames)))

            # Save as .tar.gz
            tarname = os.path.join(out_dir, '{}.tar.gz'.format(fname))
            with tarfile.open(tarname, 'w:gz') as tf:
                for frame in frames:
                    tf.add(frame)
            if verbose:
                print('Saved frames to {}'.format(tarname))

    # Save frame counts to output file
    countname = os.path.join(out_dir, '_frame_counts.txt')
    if verbose:
        print('Writing frame counts to {}'.format(countname))
    with open(countname, 'w') as countf:
        for k, v in frame_counts.items():
            countf.write('{}\t{}\n'.format(k, v))

    if verbose:
        print('Done!')

if __name__ == '__main__':

    import os
    import sys

    args = get_args();

    if not args.output:
        default = os.path.join(args.input, 'converted_frames')
        if args.verbose:
            print('No output directory specified. Using default directory {}'.format(default))
        args.output = default

    if not os.path.isdir(args.input):
        sys.exit('Input directory {} does not exist!'.format(args.input))

    if not os.path.isdir(args.output):
        if args.verbose:
            print('Creating output directory {}'.format(args.output))
        os.makedirs(args.output)

    main(args.input, args.output, args.ftype, args.verbose)
