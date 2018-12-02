from Neuron import Neuron
from NewNeuralMap import NewNeuralMap
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import InputsGetter as ig
from matplotlib import pyplot as plt

def main():
    print("Program started")

    # path = './Images/**/*.ppm'
    # inputs_mt = ig.get_all_images_vectors_mt(path, 44)
    # ig.save_inputs_to_json(inputs_mt)
    # print(len(inputs_mt))

    # inputs = ig.load_inputs_from_json()
    # print(len(inputs))
    neural_network_test()
    print("Program finished")


def neural_network_test():
    rows = 10
    cols = 10
    features = 3
    elements = 2
    cycles = 1000
    learning_rate = 0.2
    number_of_classes = elements

    #inputs = np.random.random((elements,features))

    inputs = np.array(([1,0,0],[0,0,1]))

    newmap = NewNeuralMap(rows, cols, features)

    start = timer()
    print("Learning..")
    # newmap.learn(inputs, cycles, learning_rate)
    newmap = NewNeuralMap.read_from_file()
    end = timer()
    print("Learning time: %s seconds" %(end - start))

    # newmap.print()

    newmap.build_classificator(number_of_classes)

    # newmap.safe_to_file()


def images_loading_test(min_threads = 4, max_threads = 1024):
    path = './Images/**/*.ppm'
    x = []
    y = []
    
    i = min_threads
    while i <= max_threads:
        print("%s threads running.." % (i))
        start = timer()
        inputs_mt = ig.get_all_images_vectors_mt(path, i)
        end = timer()
        
        x.append(i)
        y.append(end-start)
        i += 1


    min_time = min(y)
    optimum_number_of_threads = x[y.index(min(y))]

    print("************************")
    print("Best result:")
    print("%s threads " % optimum_number_of_threads)
    print("%s seconds" % min_time)
    print("************************")

    plt.plot(x,y)
    plt.title("Loading %s images time" % len(inputs_mt))
    plt.ylabel("Time in seconds")
    plt.xlabel("Number of threads")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
