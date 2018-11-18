import numpy as np
from Neuron import Neuron


class NeuralMap:
    def __init__(self, rows, cols, features):
        self.rows = rows
        self.cols = cols
        self.features_number = features
        self.neurons = self.create_map()


    def create_map(self):
        k = 0
        array = np.empty( (self.rows, self.cols), dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                array[i,j] = Neuron(self.features_number, k)
                k = k + 1
        return array
    

    def print_weights_as_list(self):
        for element in self.neurons.flat:
            print(element)


    def get_node_by_id(self, node_id):
        for element in self.neurons.flat:
            if element.id is node_id:
                return element


    def print_distances_from_point(self, inputs):
        for element in self.neurons.flat:
            print(element.id, ": ", str(element.distance(inputs)))
