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
    # neural_network_test()
    # images_loading_test()
    # actualMain()
    # neural_network_test()
    images_loading_test()
    print("Program finished")


def neural_network_test():
    rows = 32
    cols = 32
    features = 3
    elements = 5
    cycles = 1500
    learning_rate = 1
    number_of_classes = 3

    filename = "testmap.txt"

    # inputs = np.random.random((elements,features))

    inputs = np.array(([1,0,0],[0,0,1], [0,1,0]))

    inputs = np.vstack((inputs, [0.9,0.1,0.1]))
    inputs = np.vstack((inputs, [0.8,0.15,0.15]))
    inputs = np.vstack((inputs, [0.8,0.2,0.2]))

    inputs = np.vstack((inputs, [0.1,0.9,0.1]))
    inputs = np.vstack((inputs, [0.15,0.8,0.15]))
    inputs = np.vstack((inputs, [0.2,0.8,0.2]))

    
    inputs = np.vstack((inputs, [0.1,0.1,0.9]))
    inputs = np.vstack((inputs, [0.15,0.15,0.8]))
    inputs = np.vstack((inputs, [0.2,0.2,0.8]))

    newmap = NewNeuralMap(rows, cols, features)

    # print(inputs)

    # start = timer()
    # print("Learning..")

    # cycles = 1000
    # learning_rate = 1
    # newmap.learn(inputs, cycles, learning_rate)
    # cycles = 5000
    # learning_rate = 0.4
    # newmap.learn(inputs, cycles, learning_rate)

    # end = timer()
    # print("Learning time: %s seconds" %(end - start))
    # newmap.safe_to_file(filename)


    newmap = NewNeuralMap.read_from_file(filename)

    newmap.build_classificator_correct(number_of_classes + 2, True)
    newmap.print()
    newmap.print_classes()

    number_of_classes = np.max(newmap.classes)
    print("Found %s classes" % number_of_classes)
    # newmap.print_classes()
    # newmap.safe_to_file(filename)

def neural_network_test_read_fromfile():
    number_of_classes = 3
    filename = "testmap.txt"
    newmap = NewNeuralMap.read_from_file(filename)

    # newmap.build_classificator(number_of_classes, False)
    newmap.build_classificator_correct(number_of_classes, False)
    newmap.print()
    newmap.print_classes()

    number_of_classes = newmap.get_number_of_classes()
    print("Found %s classes" % number_of_classes)
    # newmap.print_classes()
    # newmap.safe_to_file(filename)
    print(newmap.classes)


def actualMain():
    path = './Images/**/*.ppm'
    rows = 32
    cols = 32
    features =  1024
    number_of_classes = 5
    newmap = NewNeuralMap(rows, cols, features) 

    inputs = ig.get_all_images_vectors_mt(path, 45)

    start = timer()
    print("Learning..")

    cycles = 1000
    learning_rate = 1
    newmap.learn(inputs, cycles, learning_rate)
    cycles = 3000
    learning_rate = 0.4
    newmap.learn(inputs, cycles, learning_rate)

    end = timer()
    print("Learning time: %s seconds" %(end - start))

    newmap.build_classificator_correct(number_of_classes, False)
    
    newmap.print_classes()
    number_of_classes = newmap.get_number_of_classes()
    print("Found %s classes" % number_of_classes)


def images_loading_test(min_threads = 35, max_threads = 50):
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