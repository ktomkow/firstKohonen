from NeuralMap import NeuralMap
from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from timeit import default_timer as timer

def main():
    print("Program started")

    features = 3
    elements = 100

    height = 60
    width = 60
    cycles = 100

    learn_with_threading = False

    mymap = NeuralMap(height,width,features)
    #mymap.print()

    # pattern1 = np.array((1,0,0))
    # pattern2 = np.array((0,1,0))
    # pattern3 = np.array((0,0,1))
    # pattern4 = np.array((1,1,1))
    # pattern5 = np.array((0,0,0))

    # array = np.empty(5, dtype=object)
    # array[0] = pattern1
    # array[1] = pattern2
    # array[2] = pattern3
    # array[3] = pattern4
    # array[4] = pattern5

    array = np.empty(elements, dtype=object)
    for i in range(elements):
        array[i] = np.random.random(features)

    print()
    print("Patterns: " + str(array))
    print()

    start = timer()

    if learn_with_threading:
        mymap.learn_with_threading(array, cycles)
    else: 
        mymap.learn(array, cycles)


    end = timer()
    print("Learning time: %s seconds" %(end - start))

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
                if features == 3: 
                    array[i,j] = input_data.neurons[i][j].return_weights_as_vector() * 255
                else:
                    mean = np.mean(input_data.neurons[i][j].return_weights_as_vector())
                    array[i,j] = np.array((mean, mean, mean)) * 255
        return array


    img = convert_map_to_array(mymap)

    img = Image.fromarray(img, 'RGB')
    # img.save('my.png')
    img.show()

    print("Program finished")
if __name__ == "__main__":
    main()
