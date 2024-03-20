# @toffanetto

import numpy as np
import imageio.v2 as iio

# Read image from file

image_raw = iio.imread('A/img/test_image-1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

# 1) Resizing img to 50%
#   The raw image have the size 540x960
#   Resizing to 50% of original size, we get a 270x480 image

image_reduced_50 = np.zeros([270, 480, 3], dtype=np.uint8)

print("IMG reduced 50% Tensor size: "+str(image_reduced_50.shape))

for k in range(3):
    for j in range(480):
        for i in range(270):
            aux = np.array([image_raw[i*2, j*2, k], image_raw[i*2+1, j*2, k], image_raw[i*2, j*2+1, k], image_raw[i*2+1, j*2+1, k]])
            image_reduced_50[i, j, k] = (np.average(aux))

iio.imwrite('A/img/image_reduced_50.jpg', image_reduced_50)