from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np

def main():
    print("Program started")

    mymap = NeuralMap(3,2,2)
    mymap.print_weights_as_list()

    inputs = np.array([3,4])
    mymap.print_distances_from_point(inputs)

    print("done")
    print(mymap.get_node_by_id(2))
    print(mymap.get_node_by_id(-2))

    print("Program finished")

if __name__ == "__main__":
    main()