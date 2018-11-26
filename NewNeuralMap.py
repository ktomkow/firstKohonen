import numpy as np
import numba
from timeit import default_timer as timer
from numba import jit

class NewNeuralMap:
    def __init__(self, rows, cols, features):
        self.rows = rows
        self.cols = cols
        self.features_number = features
        self.neurons = self.create_map()

    @jit(nopython=True, parallel=True)
    def create_map(self):
        array = np.random.random((self.rows, self.cols, self.features_number))
        return array