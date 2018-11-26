from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import progressbar


def main():
    print("Program started")

    features = 3
    elements = 40

    height = 40
    width = 40
    cycles1 = 100
    cycles2 = 100
    cycles3 = 600
    cycles4 = 800

    multithreading = False
    learning_rate1 = 5 # 0.01 as default
    learning_rate2 = 1.5 # 0.01 as default
    learning_rate3 = 0.95 # 0.01 as default
    learning_rate4 = 0.55 # 0.01 as default

    mymap = NeuralMap(height,width,features)
    print_map(mymap)

    # pattern1 = np.array((1,0,0))
    # array[0] = pattern1

    array = np.empty(elements, dtype=object)
    for i in range(elements):
        array[i] = np.random.random(features)

    start = timer()

    print("Part 1")
    mymap.learn(array, cycles1, multithreading, learning_rate1)
    print_map(mymap)
    print("Part 2")
    mymap.learn(array, cycles2, multithreading, learning_rate2)
    print_map(mymap)
    print("Part 3")
    mymap.learn(array, cycles3, multithreading, learning_rate3)
    print_map(mymap)
    print("Part 4")
    mymap.learn(array, cycles4, multithreading, learning_rate4)
    print_map(mymap)

    end = timer()
    print("Learning time: %s seconds" %(end - start))

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
