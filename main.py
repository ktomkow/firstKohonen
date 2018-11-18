from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Program started")

    features = 2

    mymap = NeuralMap(4,4,features)
    #mymap.print()

    pattern1 = np.array((1,0,0))
    pattern2 = np.array((0,1,0))
    pattern3 = np.array((0,0,1))

    pattern1 = np.random.random(features)
    pattern2 = np.random.random(features)
    pattern3 = np.random.random(features)

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
    print("pattern1 - Distance: " + str(np.linalg.norm(pattern1 - nearest.weights)))

    nearest = mymap.get_nearest_neuron(pattern2)
    print("Nearest for pattern: " + str(pattern2) + " : " + str(nearest))
    print("pattern2 - Distance: " + str(np.linalg.norm(pattern2 - nearest.weights)))

    nearest = mymap.get_nearest_neuron(pattern3)
    print("Nearest for pattern: " + str(pattern3) + " : " + str(nearest))
    print("pattern3 - Distance: " + str(np.linalg.norm(pattern3 - nearest.weights)))

    print("Program finished")
if __name__ == "__main__":
    main()
