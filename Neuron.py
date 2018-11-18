import numpy as np
import random


class Neuron:
    def __init__(self, length, new_id):
        self.id = new_id
        self.weights = np.zeros((length), dtype=np.float64)
        self.weights = self.assign_random_values(self.weights)


    def assign_random_values(self, weights):
        for i in range(len(weights)):
            weights[i] = random.random()
        return weights


    def distance(self, input_vector):
        array_of_differences = self.weights - input_vector
        array_of_squares = array_of_differences ** 2
        sum_of_squares = array_of_squares.sum()
        distance = sum_of_squares ** 0.5
        return distance


    def __str__(self):
        return str(str(self.id) + ": " + str(self.weights))