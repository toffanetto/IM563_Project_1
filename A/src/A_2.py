# @toffanetto

import numpy as np
import imageio.v2 as iio

def rgb2GsArray(image_array, r, g, b):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    imageGS = np.zeros([n_rows, n_collumns], dtype=np.uint8) # Empty arraw for the construction of grayscale image

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
        
            intensity = (r/100)*image_array[i, j, 0] + (g/100)*image_array[i, j, 1] + (b/100)*image_array[i, j, 2] # Sum of intensities of RGB Channels

            imageGS[i, j]  = np.uint8(intensity) # Giver for the GS pixel the respective intensity
            
    return imageGS

# Read image from file

image_raw = iio.imread('A/img/test_image-1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

image_GS_1= rgb2GsArray(image_raw, 30, 55, 15)

iio.imwrite('A/img/image_GS_1.jpg', image_GS_1)

image_GS_2= rgb2GsArray(image_raw, 30, 15, 55)

iio.imwrite('A/img/image_GS_2.jpg', image_GS_2)

image_GS_3= rgb2GsArray(image_raw, 55, 30, 15)

iio.imwrite('A/img/image_GS_3.jpg', image_GS_3)

image_GS_4= rgb2GsArray(image_raw, 15, 55, 30)

iio.imwrite('A/img/image_GS_4.jpg', image_GS_4)