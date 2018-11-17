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


def main():
    print("Program started")

    mymap = NeuralMap(3,2,2)
    mymap.print_weights_as_list()

    inputs = np.array([3,4])
    mymap.print_distances_from_point(inputs)

    print("done")
    print(mymap.get_node_by_id(2))
    print(mymap.get_node_by_id(-2))

    print("Program finished")

if __name__ == "__main__":
    main()