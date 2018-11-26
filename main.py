from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import progressbar


def main():
    print("Program started")

    features = 3
    elements = 500

    height = 100
    width = 100
    cycles = 1000

    multithreading = True # True - faster
    learning_rate = 0.1 # 0.01 as default

    mymap = NeuralMap(height,width,features)

    # pattern1 = np.array((1,0,0))
    # array[0] = pattern1

    array = np.empty(elements, dtype=object)
    for i in range(elements):
        array[i] = np.random.random(features)

    start = timer()

    mymap.learn(array, cycles, multithreading, learning_rate)

    end = timer()
    print("Learning time: %s seconds" %(end - start))

    print_map(mymap)

    print("Program finished")

def convert_map_to_array(mymap):
    height = mymap.rows
    width = mymap.cols
    features = mymap.features_number
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if features == 3: 
                array[i,j] = mymap.neurons[i][j].return_weights_as_vector() * 255
            else:
                mean = np.mean(mymap.neurons[i][j].return_weights_as_vector())
                array[i,j] = np.array((mean, mean, mean)) * 255
    return array

def print_map(mymap):
    img = convert_map_to_array(mymap)
    img = Image.fromarray(img, 'RGB')
    img.save('my.png')
    img.show()

if __name__ == "__main__":
    main()
