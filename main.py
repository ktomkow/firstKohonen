from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Program started")

    mymap = NeuralMap(20,4,2)
    mymap.print()    

    pattern1 = np.array((0.85, 0.35))
    pattern2 = np.array((0.2, 0.65))
    pattern3 = np.array((0.4, 0.4))
    array = np.empty(3, dtype=object)
    array[0] = pattern1
    array[1] = pattern2
    array[2] = pattern3

    print()
    print("Patterns: " + str(array))
    print()

    cycles = 10000
    mymap.learn(array, cycles)

    mymap.print()

    print("*****************")
    print("Results:")
    print("*****************")

    nearest = mymap.get_nearest_neuron(pattern1)
    print("Nearest for pattern: " + str(pattern1) + " : " + str(nearest))

    nearest = mymap.get_nearest_neuron(pattern2)
    print("Nearest for pattern: " + str(pattern2) + " : " + str(nearest))

    nearest = mymap.get_nearest_neuron(pattern3)
    print("Nearest for patter: " + str(pattern3) + " : " + str(nearest))

    print("Program finished")
if __name__ == "__main__":
    main()
