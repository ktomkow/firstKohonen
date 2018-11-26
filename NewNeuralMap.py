import numpy as np
import progressbar
from timeit import default_timer as timer
from PIL import Image
from random import randint

class NewNeuralMap:
    def __init__(self, rows, cols, features):
        self.rows = rows
        self.cols = cols
        self.features_number = features
        self.neurons = self.create_map()
        self.default_learning_rate = 0.01
        self.getting_winner_time = 0
        self.changin_time = 0


    def create_map(self):
        array = np.random.random((self.rows, self.cols, self.features_number))
        return array

    def learn(self, inputs_array, cycles, learning_rate = 0.01):
        bar = self.bar_create()
        bar.update(0)
        cycle_number = 0
        learning_rate = self.check_learning_rate(learning_rate)
        for i in range(cycles):
            percentage_done = (i/(cycles-1))*100
            bar.update(percentage_done)
            input_vector = inputs_array[randint(0, inputs_array.shape[0] - 1),:]
            winner_position = self.get_winner_neuron(input_vector)
            # print("Winner: %s" % winner_position)
            start = timer()
            for j in range(self.rows):
                for k in range(self.cols):
                    neuron_position = np.array((j,k))
                    distance_from_winner = np.linalg.norm(neuron_position - winner_position)
                    neighbour_ratio = np.exp(-0.693147180559945 * distance_from_winner)
                    self.neurons[j,k] = self.neurons[j,k] + learning_rate * neighbour_ratio * (input_vector - self.neurons[j,k])
            end = timer()

        print("Getting winer time: %s seconds" %(self.getting_winner_time))    
        print("Changing weights time: %s seconds" %(end - start))                     

    def get_winner_neuron(self, input_vector):
        start = timer()
        col = 0
        row = 0
        position = np.array
        minimum = 999999
        for i in range(self.rows):
            for j in range(self.cols):
                distance = np.linalg.norm(self.neurons[i,j] - input_vector)
                # print("%s %s --%s--> %s" % (np.array((i,j)), self.neurons[i,j], distance, input_vector))
                if distance < minimum:
                    row = i
                    col = j
                    minimum = distance

        end = timer()
        self.getting_winner_time += end-start
        return np.array((row, col))

    def check_learning_rate(self, learning_rate):
        if learning_rate > 0:
            return learning_rate
        else:
            if learning_rate < 0:
                return - learning_rate
            else:
                return self.default_learning_rate

    def convert_to_print(self):
        array = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.features_number == 3: 
                    array[i,j] = self.neurons[i][j] * 255
                else:
                    mean = np.mean(self.neurons[i][j])
                    array[i,j] = np.array((mean, mean, mean)) * 255
        return array
    
    def print(self):
        array = self.convert_to_print()
        img = Image.fromarray(array, 'RGB')
        img.show()

    def print_weights(self):
        print("Weights:")
        print(self.neurons)


    def bar_create(self):
        bar = progressbar.ProgressBar(maxval=100, \
            widgets=[progressbar.Bar('#', '', '', '-'), ' ', progressbar.Percentage()])
        bar.start()
        return bar

    def bar_update(self, bar, percentage_done):
        bar.update(percentage_done)


