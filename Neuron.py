import numpy as np
import random


class Neuron:
    def __init__(self, length, new_id):
        self.id = new_id
        self.init_weights(length)


    def init_weights(self, length):
        self.weights = self.random_weights(length)


    def random_weights(self, length):
        weights = np.random.random(length)
        return weights


    def distance(self, input_vector):
        return np.linalg.norm(self.weights - input_vector)


    def __str__(self):
        return str(str(self.id) + ": " + str(self.weights))