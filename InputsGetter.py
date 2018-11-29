import ImageProcessor as ip
import numpy as np
import os

def get_all_images_vectors():

    path = './Images'
    inputs = np.ones((1024)) # to initialize array

    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            vector = ip.get_normalized_vector((os.path.join(root,filename)))
            inputs = np.vstack((inputs, vector))

    inputs = inputs[1:] # to remove the first row
    return inputs