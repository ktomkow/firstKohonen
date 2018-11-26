import numpy as np
import progressbar
from timeit import default_timer as timer
from PIL import Image


class NewNeuralMap:
    def __init__(self, rows, cols, features):
        self.rows = rows
        self.cols = cols
        self.features_number = features
        self.neurons = self.create_map()


    def create_map(self):
        array = np.random.random((self.rows, self.cols, self.features_number))
        return array

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

    def bar_create(self):
        bar = progressbar.ProgressBar(maxval=100, \
            widgets=[progressbar.Bar('#', '', '', '-'), ' ', progressbar.Percentage()])
        bar.start()
        return bar

    def bar_update(self, bar, iter, cycles, percentage_left):
        if iter % (percentage_left) == 0:
            percentage_completed = 100 - ((cycles - iter)/cycles * 100)
            bar.update(percentage_completed)


