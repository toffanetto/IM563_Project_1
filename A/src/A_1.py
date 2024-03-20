# @toffanetto

import numpy as np
import imageio.v2 as iio

def imageArrayReduce(image_array, reduce_ratio):
    n_rows = np.uint16(image_array.shape[0]*(reduce_ratio/100))
    n_collumns = np.uint16(image_array.shape[1]*(reduce_ratio/100))

    image_reduced = np.zeros([n_rows, n_collumns, 3], dtype=np.uint8)

    ratio_step = np.uint16(1/(reduce_ratio/100))

    for k in range(image_array.shape[2]):
        for j in range(n_collumns):
            for i in range(n_rows):

                intensity = 0

                for m in range(ratio_step):
                    for n in range(ratio_step):
                        intensity += image_array[i*ratio_step + m, j*ratio_step + n, k]

                image_reduced[i, j, k]  = np.uint8(intensity/(ratio_step*ratio_step))
                
    return image_reduced

# Read image from file

image_raw = iio.imread('A/img/test_image-1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

# 1) Resizing img to 50%
#   The raw image have the size 540x960
#   Resizing to 50% of original size, we get a 270x480 image

image_reduced_50 = imageArrayReduce(image_raw, 50)

print("IMG reduced 50% Tensor size: "+str(image_reduced_50.shape))

iio.imwrite('A/img/image_reduced_50.jpg', image_reduced_50)

# 1) Resizing img to 25%
#   The raw image have the size 540x960
#   Resizing to 25% of original size, we get a 135x240 image

image_reduced_25 = imageArrayReduce(image_raw, 25)

print("IMG reduced 50% Tensor size: "+str(image_reduced_25.shape))

iio.imwrite('A/img/image_reduced_25.jpg', image_reduced_25)

