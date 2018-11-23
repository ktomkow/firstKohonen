from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    print("Program started")

    features = 3

    height = 16
    width = 16

    mymap = NeuralMap(height,width,features)
    #mymap.print()

    pattern1 = np.array((1,0,0))
    pattern2 = np.array((0,1,0))
    pattern3 = np.array((0,0,1))
    pattern4 = np.array((1,1,1))
    pattern5 = np.array((0,0,0))
    # pattern1 = np.random.random(features)
    # pattern2 = np.random.random(features)
    # pattern3 = np.random.random(features)

    array = np.empty(5, dtype=object)
    array[0] = pattern1
    array[1] = pattern2
    array[2] = pattern3
    array[3] = pattern4
    array[4] = pattern5

    print()
    print("Patterns: " + str(array))
    print()

    cycles = 4000
    mymap.learn(array, cycles)

    # mymap.print()

    # print("*****************")
    # print("Results:")
    # print("*****************")

    # nearest = mymap.get_nearest_neuron(pattern1)
    # print("Nearest for pattern: " + str(pattern1) + " : " + str(nearest))
    # print("pattern1 - Distance: " + str(np.linalg.norm(pattern1 - nearest.weights)))

    # nearest = mymap.get_nearest_neuron(pattern2)
    # print("Nearest for pattern: " + str(pattern2) + " : " + str(nearest))
    # print("pattern2 - Distance: " + str(np.linalg.norm(pattern2 - nearest.weights)))

    # nearest = mymap.get_nearest_neuron(pattern3)
    # print("Nearest for pattern: " + str(pattern3) + " : " + str(nearest))
    # print("pattern3 - Distance: " + str(np.linalg.norm(pattern3 - nearest.weights)))

    height = mymap.rows
    width = mymap.cols

    def convert_map_to_array(input_data):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                array[i,j] = input_data.neurons[i][j].return_weights_as_vector() * 255
        return array


    img = convert_map_to_array(mymap)

    img = Image.fromarray(img, 'RGB')
    img.save('my.png')
    img.show()

    print("Program finished")
if __name__ == "__main__":
    main()
