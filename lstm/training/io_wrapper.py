"""Wrapper for accessing files locally or through GCS"""

from tensorflow import gfile
import time

# Number of seconds to wait after failed GCS access
WAIT = 5

# Number of tries before progagating thrown exception
TRY = 10

def _wait_loop(f):
    '''Continuously attempt to call f

    Retries after any exception thrown, assuming GCS UnavailableError.
    Propagates thrown exception after TRY tries.

    Input:
        f: funciton; callable with no arguments

    Output:
        Return value of f
    '''

    for i in range(1, TRY + 1):
        try:
            return f()
        except Exception as e:
            print('IO Wrapper exception in `{}` [{}]:\n{}\n{}'.format(f.__name__, i, e.__doc__, e.message))
            if i < TRY:
                print('Waiting {} seconds...'.format(WAIT))
                time.sleep(WAIT)
            else:
                raise e

def exists(path):
    '''Check if path exists on disk or GCS

    Input:
        path: str; path to file or directory

    Output:
        True if path exists, False otherwise
    '''

    def exists_wrapper():
        return gfile.Exists(path)

    return _wait_loop(exists_wrapper)

def verify_dirs_exist(dirname):
    '''Verify that the directory exists

    Will recursively create directories as needed.

    Input:
        dirname: str; directory name to create
    '''

    if not exists(dirname):
        gfile.MakeDirs(dirname)
