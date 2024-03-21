# @toffanetto

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

def borderDetector(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_borders = np.zeros([n_rows, n_collumns, 3], dtype=np.uint8)

    # Get borders from the image....

    return image_borders

image_raw = iio.imread('A/img/image_GS_1_equalized.jpg')

print("IMG Tensor size: "+str(image_raw.shape))