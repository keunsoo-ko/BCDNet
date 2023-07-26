from os.path import join
from lib.dataset import TestDatasetFromFolder
import glob, os
import numpy as np

class Data():
    def __init__(self, path):
        self.test_dir = np.concatenate(
            [glob.glob(join(path, _, '*.*')) \
             for _ in os.listdir(path) if 'Noise' in _])

    def get_test_set(self):
        return TestDatasetFromFolder(self.test_dir)
