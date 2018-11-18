from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Program started")

    mymap = NeuralMap(3,3,2)
    mymap.print_weights_as_list()

    inputs = np.array([3,4])
    mymap.print_distances_from_point(inputs)

    print("*****************")

    mymap.print_structure()

    print("*****************")

    nearest = mymap.get_nearest_neuron(inputs)
    print("Nearest: " + str(nearest))

    print("Program finished")

    mymap = NeuralMap(4,4,2)
    mymap.print()    

    pattern1 = np.array((1, 1))
    pattern2 = np.array((0.1, 0.1))
    array = np.empty(2, dtype=object)
    array[0] = pattern1
    array[1] = pattern2

    print()
    print("Patterns: " + str(array))
    print()

    cycles = 1000
    mymap.learn(array, cycles)

    mymap.print()

    print("*****************")
    print("Results:")
    print("*****************")

    nearest = mymap.get_nearest_neuron(pattern1)
    print("Nearest for pattern: " + str(pattern1) + " : " + str(nearest))

    nearest = mymap.get_nearest_neuron(pattern2)
    print("Nearest for pattern: " + str(pattern2) + " : " + str(nearest))
if __name__ == "__main__":
    main()