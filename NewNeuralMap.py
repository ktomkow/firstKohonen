import numpy as np
import progressbar
from timeit import default_timer as timer
from PIL import Image
from random import randint
import threading
import jsonpickle
import math


class NewNeuralMap:
    def __init__(self, rows, cols, features):
        self.rows = rows
        self.cols = cols
        self.features_number = features
        self.number_of_classes = 0
        self.neurons = self.create_map()
        self.classes = self.create_classes_array()
        self.default_learning_rate = 0.01
        self.getting_winner_time = 0
        self.changin_time = 0


    def create_map(self):
        array = np.random.random((self.rows, self.cols, self.features_number))
        return array


    def safe_to_file(self, filename = "my_map.txt"):
        frozen = jsonpickle.encode(self)
        f = open(filename, "w")
        f.write(frozen)


    @staticmethod
    def read_from_file(filename = "my_map.txt"):
        f = open(filename, "r")
        frozen = f.read()
        return jsonpickle.decode(frozen)

    
    def create_classes_array(self):
        array = np.zeros((self.rows, self.cols))
        return array

    def build_classifier(self, wanted_number_of_classes, printing = False):
        self.create_centers_of_classes(wanted_number_of_classes)   # done
        self.classify_unclassified_neurons(printing) # done 

        self.decrease_number_of_classes(wanted_number_of_classes, printing) 

        self.shuffle_classes() # done

    def create_centers_of_classes(self, wanted_number_of_classes, max_diff = 0.001):
        self.classes = self.create_classes_array()
        neighbors_classes = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.classes[i,j] == 0:
                    neighbors = self.get_neighbors(i, j)
                    numberOfSimilarNeurons = 0
                    for k in range(len(neighbors)):
                        difference = np.linalg.norm(self.neurons[i,j] - self.neurons[(neighbors[k][0]),(neighbors[k][1])])
                        if difference <= max_diff:
                            numberOfSimilarNeurons += 1

                    if numberOfSimilarNeurons >= math.ceil(len(neighbors) * 0.7):
                        neighbors_classes = []
                        for k in range(len(neighbors)):
                            if self.classes[(neighbors[k][0]),(neighbors[k][1])] != 0:
                                neighbors_classes.append(self.classes[(neighbors[k][0]),(neighbors[k][1])])

                        how_many_different_classes = len(set(neighbors_classes))

                        if how_many_different_classes == 1:
                            c = set(neighbors_classes).pop()
                            self.classes[i,j] = int(c)
                            for k in range(len(neighbors)):
                                self.classes[(neighbors[k][0]),(neighbors[k][1])] = int(c)

                        if how_many_different_classes == 0:
                            c = np.max(self.classes) + 1
                            self.classes[i,j] = int(c)
                            for k in range(len(neighbors)):
                                self.classes[(neighbors[k][0]),(neighbors[k][1])] = int(c)


        number_of_classes = np.max(self.classes)
        if(number_of_classes < wanted_number_of_classes):
            self.create_centers_of_classes(wanted_number_of_classes, max_diff + 0.001)
        else:
            return

    def classify_unclassified_neurons(self, printing = False):
        while self.number_of_unclassified_neurons() != 0:
            class_proposal = self.create_classes_array()
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i,j] == 0:
                        neuron_position = np.array((i,j))
                        neighbors = self.get_neighbors(i, j)
                        neighbors_classes = []
                        for k in range(len(neighbors)):
                            if self.classes[(neighbors[k][0]),(neighbors[k][1])] != 0:
                                neighbors_classes.append(self.classes[(neighbors[k][0]),(neighbors[k][1])])

                        neighbors_classes = list(set(neighbors_classes))
                        dictionary = {}
                        for k in range(len(neighbors_classes)):
                            dictionary[neighbors_classes[k]] = 0
                        
                        for k in range(len(neighbors)):
                            neighbor_class = self.classes[(neighbors[k][0]),(neighbors[k][1])]
                            if neighbor_class != 0:
                                dictionary[neighbor_class] += 1
            
                        chosen_class = 0
                        if len(dictionary) != 0: # if dictionary not empty
                            chosen_class = max(dictionary.keys(), key=(lambda k: dictionary[k]))
                            class_proposal[i,j] = chosen_class

            for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i,j] == 0:
                        self.classes[i,j] = int(class_proposal[i,j])

            if printing:
                self.print_classes()

    def decrease_number_of_classes(self, wanted_number_of_classes, printing = False):
        while(self.get_number_of_classes() > wanted_number_of_classes):
            actual_number_of_classes = self.get_number_of_classes()
            min_difference = 99999999999
            firstClassToMerge = 0
            secondClassToMerge = 0
            for i in range(actual_number_of_classes):
                firstClassAverageNeuron = self.get_average_neuron_of_class(i)
                if np.isnan(firstClassAverageNeuron).any():
                    continue

                for j in range(actual_number_of_classes):
                    if(i == j or j == 0 or j == 0):
                        continue

                    secondClassAverageNeuron = self.get_average_neuron_of_class(j)
                    if np.isnan(secondClassAverageNeuron).any():
                        continue

                    difference = np.linalg.norm(firstClassAverageNeuron - secondClassAverageNeuron) 
                    difference = difference ** 2

                    if difference < min_difference:
                        min_difference = difference
                        firstClassToMerge = i
                        secondClassToMerge = j

            self.merge_classes(firstClassToMerge, secondClassToMerge)
            if(printing):
                self.print_classes()

    def get_number_of_classes(self):
        return int(np.max(self.classes))


    def merge_classes(self, classToMergeInto, classToRemove):
        for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i,j] == classToRemove:
                        self.classes[i,j] = classToMergeInto
        self.move_classes(classToRemove)


    def move_classes(self, removed_class):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.classes[i,j] >= removed_class:
                    self.classes[i,j] = self.classes[i,j] - 1


    # deprecated
    def build_classificator(self, number_of_classes, printing = False):
        self.number_of_classes = number_of_classes
        n3 = 0
        n5 = 0
        n8 = 0
        max_diff = 0.08 # nice
        self.classes = self.create_classes_array()
        neighbors_classes = []
        # step 1 - finding really similar neurons right next to neuron and group them

        for i in range(self.rows):
            for j in range(self.cols):
                if self.classes[i,j] == 0:
                    neuron_position = np.array((i,j))
                    neighbors = self.get_neighbors(i, j)
                    # n = 0
                    numberOfSimilarNeurons = 0
                    for k in range(len(neighbors)):
                        difference = np.linalg.norm(self.neurons[i,j] - self.neurons[(neighbors[k][0]),(neighbors[k][1])])
                        if difference <= max_diff:
                            # n += 1
                            numberOfSimilarNeurons += 1

                    # if n >= math.ceil(len(neighbors) * 0.7):
                    if numberOfSimilarNeurons >= math.ceil(len(neighbors) * 0.7):
                        neighbors_classes = []
                        for k in range(len(neighbors)):
                            if self.classes[(neighbors[k][0]),(neighbors[k][1])] != 0:
                                neighbors_classes.append(self.classes[(neighbors[k][0]),(neighbors[k][1])])

                        how_many_different_classes = len(set(neighbors_classes))

                        if how_many_different_classes == 1:
                            c = set(neighbors_classes).pop()
                            self.classes[i,j] = c
                            for k in range(len(neighbors)):
                                self.classes[(neighbors[k][0]),(neighbors[k][1])] = (int)(c)

                        if how_many_different_classes == 0:
                            c = np.max(self.classes) + 1
                            self.classes[i,j] = c
                            for k in range(len(neighbors)):
                                self.classes[(neighbors[k][0]),(neighbors[k][1])] = (int)(c)

        #step 1 completed
        if printing:
            self.print_classes()
        #step 2 k nearest neighbors - find all neighbors of a neuron and classify it to the same class as most of the neurons
        while self.number_of_unclassified_neurons() != 0:
        # for l in range(3):
            class_proposal = self.create_classes_array()
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i,j] == 0:
                        neuron_position = np.array((i,j))
                        neighbors = self.get_neighbors(i, j)
                        neighbors_classes = []
                        for k in range(len(neighbors)):
                            if self.classes[(neighbors[k][0]),(neighbors[k][1])] != 0:
                                neighbors_classes.append(self.classes[(neighbors[k][0]),(neighbors[k][1])])

                        neighbors_classes = list(set(neighbors_classes))
                        dictionary = {}
                        for k in range(len(neighbors_classes)):
                            dictionary[neighbors_classes[k]] = 0
                        
                        for k in range(len(neighbors)):
                            neighbor_class = self.classes[(neighbors[k][0]),(neighbors[k][1])]
                            if neighbor_class != 0:
                                dictionary[neighbor_class] += 1
            
                        chosen_class = 0
                        if len(dictionary) != 0: # if dictionary not empty
                            chosen_class = max(dictionary.keys(), key=(lambda k: dictionary[k]))
                            class_proposal[i,j] = chosen_class

            for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i,j] == 0:
                        self.classes[i,j] = (int)(class_proposal[i,j])

            if printing:
                self.print_classes()

        self.shuffle_classes()
    

    def shuffle_classes(self):
        number_of_classes = np.max(self.classes)
        for k in range((int)(number_of_classes)):

            row = randint(0, self.rows - 1)
            col = randint(0, self.cols - 1)
            class_of_first_random_neuron = self.classes[row, col]
            class_of_second_random_neuron = class_of_first_random_neuron

            while class_of_second_random_neuron == class_of_first_random_neuron:
                row = randint(0, self.rows - 1)
                col = randint(0, self.cols - 1)
                class_of_second_random_neuron = self.classes[row, col]
            
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.classes[i, j] == class_of_first_random_neuron:
                        self.classes[i, j] = class_of_second_random_neuron

                    elif self.classes[i, j] == class_of_second_random_neuron:
                        self.classes[i, j] = class_of_first_random_neuron


    def get_neurons_of_class(self, looking_class):
        neurons = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.classes[i,j] == looking_class:
                    neurons.append((i,j))
        return neurons


    def get_average_neuron_of_class(self, looking_class):
        average_neuron = np.zeros(self.features_number)
        neurons = self.get_neurons_of_class(looking_class)
        number_of_neurons = len(neurons)
        for i in range(number_of_neurons):
            current_neuron = self.neurons[neurons[i][0], neurons[i][1]]
            average_neuron = average_neuron + current_neuron
        return average_neuron/float(number_of_neurons)


    def number_of_unclassified_neurons(self):
        n = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.classes[i,j] == 0:
                    n += 1
        return n

    def row_out_of_range(self, row):
        if row < 0:
            return True
        if row > self.rows - 1:
            return True
        return False

    def col_out_of_range(self, col):
        if col < 0:
            return True
        if col > self.cols - 1:
            return True
        return False

    def get_neighbors(self, row, col): # it works!
        neighbors = []
        i = row - 1
        while i <= row + 1:
            j = col - 1
            while j <= col + 1:
                if not (i == row and j == col):
                    if not (self.row_out_of_range(i) or self.col_out_of_range(j)):
                        neighbors.append((i,j))
                j += 1
            i += 1
        return neighbors

    def cycles(self, first_row, last_row, winner_position, input_vector, learning_rate):
        j = first_row
        while j < last_row:
            for k in range(self.cols):
                neuron_position = np.array((j,k))
                distance_from_winner = np.linalg.norm(neuron_position - winner_position)
                # print("Distance from winner: %s" % distance_from_winner)
                neighbour_ratio = np.exp(-0.693147180559945 * distance_from_winner)
                self.neurons[j,k] = self.neurons[j,k] + learning_rate * neighbour_ratio * (input_vector - self.neurons[j,k])
            j += 1

    def learn(self, inputs_array, cycles, learning_rate = 0.01):
        learning_rate = self.check_learning_rate(learning_rate)
        print("Learning in %s cycles with learning rate: %s" % (cycles, learning_rate))
        bar = self.bar_create()
        bar.update(0)
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
                    # print("Distance from winner: %s" % distance_from_winner)
                    neighbour_ratio = np.exp(-0.693147180559945 * distance_from_winner)
                    self.neurons[j,k] = self.neurons[j,k] + learning_rate * neighbour_ratio * (input_vector - self.neurons[j,k])
            end = timer()
            self.changin_time +=  end - start

        print()
        print("Getting winner time: %s seconds" %(self.getting_winner_time))    
        print("Changing weights time: %s seconds" %(self.changin_time))                     

    def get_winner_neuron(self, input_vector):
        start = timer()
        col = 0
        row = 0
        minimum = 999999
        for i in range(self.rows):
            for j in range(self.cols):
                difference = np.linalg.norm(self.neurons[i,j] - input_vector)
                # print("%s %s --%s--> %s" % (np.array((i,j)), self.neurons[i,j], difference, input_vector))
                if difference < minimum:
                    row = i
                    col = j
                    minimum = difference

        end = timer()
        self.getting_winner_time += end-start
        return np.array((row, col))


    def get_winner_class(self, input_vector):
        winner_neuron = self.get_winner_neuron(input_vector)
        return self.classes[winner_neuron[0], winner_neuron[1]]

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
        img = img.resize((300,300))
        img.show()

    def print_weights(self):
        print("Weights:")
        print(self.neurons)

    
    def print_classes(self):
        max_class_number = np.max(self.classes)

        array = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.classes[i][j]
                array[i,j] = np.array((value, value, value)) * 255 / (max_class_number + 1)

        img = Image.fromarray(array, 'RGB')
        img = img.resize((300,300))
        img.show()

    def bar_create(self):
        bar = progressbar.ProgressBar(maxval=100, \
            widgets=[progressbar.Bar('#', '', '', '-'), ' ', progressbar.Percentage()])
        bar.start()
        return bar

    def bar_update(self, bar, percentage_done):
        bar.update(percentage_done)