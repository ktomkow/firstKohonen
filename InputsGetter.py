import ImageProcessor as ip
import numpy as np
import os
import progressbar
from glob import glob
import threading
import jsonpickle

def save_inputs_to_json(inputs):
    filename = "first_inputs.txt"
    frozen = jsonpickle.encode(inputs)
    f = open(filename, "w")
    f.write(frozen)


def load_inputs_from_json():
    filename = "first_inputs.txt"
    f = open(filename, "r")
    frozen = f.read()
    return jsonpickle.decode(frozen)
    

def get_all_images_vectors(path = './Images/**/*.ppm'):
    print("Loading images..")
    bar = bar_create(get_number_of_files(path))
    filepaths = get_array_of_filepaths(path)

    inputs = np.ones((1024)) # to initialize array

    inputs = np.vstack((inputs, load_images_vectors(filepaths, bar)))

    inputs = inputs[1:] # to remove the first row

    bar.finish()
    print("Loading images done")
    return inputs


def get_all_images_vectors_mt(path = './Images/**/*.ppm', number_of_threads = 24): # 44 optimum
    print("Loading images by many threads..")
    filepaths = get_array_of_filepaths(path)

    bar = bar_create(number_of_threads)
    j = 0

    threads = []
    inputs_from_threads = [None] * number_of_threads

    inputs = np.ones((1024)) # to initialize array

    filepaths_array = split_list(filepaths, number_of_threads)
        
    for i in range(number_of_threads):
        inputs_from_threads[i] = np.ones((1024)) # to initialize array
        t = threading.Thread(target=load_images_vectors_mt, args=(filepaths_array[i], inputs_from_threads, i))
        threads.append(t)
        t.start()

    for i in range(number_of_threads):
        threads[i].join()

    for i in range(number_of_threads):
        inputs = np.vstack((inputs, inputs_from_threads[i][1:]))
        j += 1
        bar_update(bar, j)

    inputs = inputs[1:] # to remove the first row
    bar.finish()
    print("Loading images done")
    return inputs


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def load_images_vectors(filepaths, bar):
    i = 0
    inputs = np.ones((1024)) # to initialize array
    for filepath in filepaths:
        vector = ip.get_normalized_vector(filepath)
        inputs = np.vstack((inputs, vector))
        i += 1
        bar_update(bar, i)

    inputs = inputs[1:] # to remove the first row
    return inputs


def load_images_vectors_mt(filepaths, inputs, i):
    for filepath in filepaths:
        vector = ip.get_normalized_vector(filepath)
        inputs[i] = np.vstack((inputs[i], vector))
    


def get_array_of_filepaths(path = './Images/**/*.ppm'):
    filepaths = []
    for filepath in glob(path, recursive=True):
        filepaths.append(filepath)

    return filepaths


def get_number_of_files(path):
    i = 0
    for filepath in glob(path, recursive=True):
        i += 1

    return i


def bar_create(number_of_files):
    bar = progressbar.ProgressBar(maxval = number_of_files, \
    widgets=[progressbar.Bar('#', '', '', '-'), ' ', progressbar.Percentage()])
    bar.start()
    return bar


def bar_update(bar, done):
    bar.update(done)