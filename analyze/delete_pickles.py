import os
import glob
import time
from data_analysis.analyze.config import pickles_path


# Delete pickles saved by the code that are older than x hours.

def remove_older_than(hours):
    now = time.time()
    for path in glob.glob(f"{pickles_path}/**", recursive=True):
        if os.path.isfile(path):
            if os.stat(path).st_mtime < now - 3600 * float(hours):
                os.remove(path)


if __name__ == '__main__':
    remove_older_than(0)
