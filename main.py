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

    height = 50
    width = 50
    cycles = 100

    multithreading = True
    learning_rate = 0.1 # 0.01 as default

    mymap = NeuralMap(height,width,features)

    # pattern1 = np.array((1,0,0))
    # array[0] = pattern1

    array = np.empty(elements, dtype=object)
    for i in range(elements):
        array[i] = np.random.random(features)

    start = timer()

    mymap.learn(array, cycles, multithreading, learning_rate)

    end = timer()
    print("Learning time: %s seconds" %(end - start))

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
