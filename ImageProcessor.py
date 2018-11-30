from PIL import Image
import numpy as np

def get_processes_image(file_path):
    image = get_image(file_path)
    image = process_image(image)
    return image


def get_normalized_vector(file_path):
    image = get_processes_image(file_path)
    return to_normalized_vector(image)


def get_image(file_path):
    image = Image.open(file_path)
    return image


def process_image(image):
    image = image.convert(mode='L')
    image =  make_square(image)
    image = image.resize((32,32))
    return image


def to_normalized_vector(image):
    array2D = np.array(image,'f') / 255
    vector = np.concatenate(array2D)
    return vector


def to_vector(image):
    array2D = np.array(image,'f')
    vector = np.concatenate(array2D)
    return vector


def make_square(image):
    width, height = image.size

    if width != height:
        if width > height:
            new_size = height
        else:
            new_size = width

        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2

        image = image.crop((left, top, right, bottom))

    return image