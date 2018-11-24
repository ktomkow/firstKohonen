import numpy as np
from Neuron import Neuron
import threading

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


    def print_structure(self):
        for i in range(self.rows):
            ids = ""
            for j in range(self.cols):
                ids = ids + " " + str(self.neurons[i,j].id)
            print(ids)


    def print(self):
        for i in range(self.rows):
            ids = ""
            for j in range(self.cols):
                ids = ids + " " + str(self.neurons[i,j])
            print(ids)


    def learn(self, inputs_array, cycles, print_status = True, learning_rate = 0.01):
        if print_status == True:
            s = 0
            percentage = 0.01 # 0.01 means 1%, 0.1 means 10% etc
            percentage_to_number = cycles * percentage
        for i in range(cycles):
            if print_status == True:
                if s % (percentage_to_number) == 0:
                    percentage_completed = 100 - ((cycles - s)/cycles * 100)
                    print(str(percentage_completed) + "% completed")
                s = s + 1

            
            inputs = np.random.choice(inputs_array)
            winner = self.get_nearest_neuron(inputs)
            winner_position = self.get_position(winner)
            for i in range(self.rows):
                self.change(i, winner_position, inputs, learning_rate)

    def learn_with_threading(self, inputs_array, cycles, print_status = True, learning_rate = 0.01):
        if print_status == True:
            s = 0
            percentage = 0.01 # 0.01 means 1%, 0.1 means 10% etc
            percentage_to_number = cycles * percentage
        for i in range(cycles):
            if print_status == True:
                if s % (percentage_to_number) == 0:
                    percentage_completed = 100 - ((cycles - s)/cycles * 100)
                    print(str(percentage_completed) + "% completed")
                s = s + 1

            inputs = np.random.choice(inputs_array)
            winner = self.get_nearest_neuron(inputs)
            winner_position = self.get_position(winner)

            threads = []
            for i in range(self.rows):
                t = threading.Thread(target=self.change, args=(i,winner_position,inputs, learning_rate))
                threads.append(t)
                t.start()
                    
            for i in range(self.rows):
                threads[i].join()


    def change(self, row, winner_position, inputs, learning_rate):
        for j in range(self.cols):
            neuron_position = np.array((row,j))
            distance_from_winner = np.linalg.norm(neuron_position - winner_position)
            neighbour_ratio = np.exp(-0.693147180559945 * distance_from_winner)
            self.neurons[row,j].correct(inputs, learning_rate, neighbour_ratio)


    def get_position(self, neuron):
        for i in range(self.rows):
                for j in range(self.cols):
                    if neuron is self.neurons[i,j]:
                        return np.array((i,j))

    def get_nearest_neuron(self, inputs):
        index = self.get_nearest_neuron_index(inputs)
        return np.take(self.neurons, index)


    def get_nearest_neuron_index(self, inputs):
        return min(range(len(self.neurons.flat)), key=lambda i: np.take(self.neurons,i).distance(inputs))


    def print_distances_from_point(self, inputs):
        for element in self.neurons.flat:
            print(element.id, ": ", str(element.distance(inputs)))
