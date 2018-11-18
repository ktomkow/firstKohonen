import numpy as np
import random


class Neuron:
    def __init__(self, length, new_id = 0):
        self.id = new_id
        self.init_weights(length)
        

    def correct(self, inputs, learning_rate = 0.01, neighbour_ratio = 1):   #neighbour_ratio -> float 0..1 -> closer to 1 means closer to winner
        self.weights = self.weights + learning_rate*neighbour_ratio*(inputs - self.weights)
    

    def init_weights(self, length):
        self.weights = self.random_weights(length)


    def random_weights(self, length):
        weights = np.random.random(length)
        return weights


    def distance(self, input_vector):
        return np.linalg.norm(self.weights - input_vector)


    def __str__(self):
        return str(str(self.id) + ": " + str(self.weights))