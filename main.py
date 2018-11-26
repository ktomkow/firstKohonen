from NeuralMap import NeuralMap
from Neuron import Neuron
from NewNeuralMap import NewNeuralMap
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import progressbar
import threading

def main():
    print("Program started")
    rows = 24
    cols = 24
    features = 3
    start = timer()
    cycles = 100
    learning_rate = 0.9

    elements = 4

    inputs = np.random.random((elements,features))
    # print(inputs)

    newmap1 = NewNeuralMap(rows, cols, features)
    newmap2 = NewNeuralMap(rows, cols, features)
    # newmap.print_weights()
    end = timer()

    print("Creating new map time: %s seconds" %(end - start))

    start = timer()
    newmap1.learn(inputs, cycles, learning_rate)
    end = timer()
    print("Learning time ONETHREAD: %s seconds" %(end - start))

    start = timer()
    newmap2.learn_mt(inputs, cycles, learning_rate)
    threads = []
    how_many_times = 12
    for i in range(how_many_times):
        t = threading.Thread(target=learn, args=(newmap2, inputs, cycles, learning_rate))
        threads.append(t)
        t.start()
            
    for i in range(how_many_times):
        threads[i].join()
    end = timer()
    print("Learning time MULTITHREAD: %s seconds" %(end - start))

    start = timer()
    newmap1.print()
    newmap2.print()
    end = timer()

    print("Printing new map time: %s seconds" %(end - start))

def learn(net,inputs, cycles, learning_rate):
    net.learn_mt(inputs, cycles, learning_rate)

def main1():
    print("Program started")

    features = 3
    elements = 40

    height = 40
    width = 40
    cycles = 1000

    multithreading = False
    learning_rate = 0.5 # 0.01 as default


    mymap = NeuralMap(height,width,features)
    print_map(mymap)

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
    # img.save('my.png')
    img.show()

if __name__ == "__main__":
    main()
