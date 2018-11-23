import numpy as np
import random


class Neuron:
    def __init__(self, length, new_id = 0):
        self.id = new_id
        self.features_number = length
        # self.init_weights()
        self.randomize_weights()
        

    def init_weights(self):
        self.weights = np.zeros(self.features_number)

    def return_weights_as_vector(self):
        return self.weights

    def randomize_weights(self):
        self.weights = np.random.random(self.features_number)
        

    def distance(self, input_vector):
        return np.linalg.norm(self.weights - input_vector)


    def correct(self, inputs, learning_rate = 0.01, neighbour_ratio = 1):   #neighbour_ratio -> float 0..1 -> closer to 1 means closer to winner
        self.weights = self.weights + learning_rate * neighbour_ratio * (inputs - self.weights)
    

    def __str__(self):
        return str(str(self.id) + ": " + str(self.weights))