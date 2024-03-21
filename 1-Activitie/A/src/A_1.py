# @toffanetto

import numpy as np
import imageio.v2 as iio

def imageArrayReduce(image_array, reduce_ratio):
    n_rows = np.uint16(image_array.shape[0]*(reduce_ratio/100)) # Number of rows of the reduced image
    n_collumns = np.uint16(image_array.shape[1]*(reduce_ratio/100)) # Number of collumns of the reduced image

    image_reduced = np.zeros([n_rows, n_collumns, 3], dtype=np.uint8) # Empty arraw for the construction of reduced image

    ratio_step = np.uint16(1/(reduce_ratio/100)) # Reduction ratio between original and reduced image

    for k in range(image_array.shape[2]): # RGB channel sweep
        for j in range(n_collumns): # Collumn sweep
            for i in range(n_rows): # Row sweep

                intensity = 0 # Initialization of the sum of intensities variable

                for m in range(ratio_step): # Sweeping the original image macropixel in vertical direction
                    for n in range(ratio_step): # Sweeping the original image macropixel in horizontal direction
                        intensity += image_array[i*ratio_step + m, j*ratio_step + n, k] # Sum of intensities of pixels in macropixel

                image_reduced[i, j, k]  = np.uint8(intensity/(ratio_step*ratio_step)) # Average of intensities, given the new pixel of reduced image
                
    return image_reduced

# Read image from file

image_raw = iio.imread('A/img/test_image-1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

# 1) Resizing image
#       50% reducing
#       The raw image have the size 540x960
#       Resizing to 50% of original size, we get a 270x480 image

image_reduced_50 = imageArrayReduce(image_raw, 50)

print("IMG reduced 50% Tensor size: "+str(image_reduced_50.shape))

iio.imwrite('A/img/image_reduced_50.jpg', image_reduced_50)

#       25% reducing
#       The raw image have the size 540x960
#       Resizing to 25% of original size, we get a 135x240 image

image_reduced_25 = imageArrayReduce(image_raw, 25)

print("IMG reduced 25% Tensor size: "+str(image_reduced_25.shape))

iio.imwrite('A/img/image_reduced_25.jpg', image_reduced_25)

