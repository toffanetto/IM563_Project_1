# @toffanetto

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

BORDER_THRESHOLD = 45
N_BORDER_SEARCH = 2
M_BORDER_SEARCH = 2

def borderDetector(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_borders = np.zeros([n_rows, n_collumns], dtype=np.uint8)
    
    for i in range(n_rows):
        for j in range(n_collumns):
            border_detect = 0
            n_pixel = 0
            for n in range(N_BORDER_SEARCH*2+1):
                if(i-N_BORDER_SEARCH+n >= 0) and (i-N_BORDER_SEARCH+n < n_rows):
                    for m in range(M_BORDER_SEARCH*2+1):
                        if(j-M_BORDER_SEARCH+m >= 0) and (j-M_BORDER_SEARCH+m < n_collumns):
                            border_detect += np.abs(image_array[i,j]-image_array[i-N_BORDER_SEARCH+n, j-M_BORDER_SEARCH+m])
                            n_pixel += 1

            if(np.sum(border_detect)/(n_pixel*3) > BORDER_THRESHOLD):
                image_borders[i, j] = 255

        print(str(np.uint8(i/n_rows*100))+'% completed')
    return image_borders

image_blur = iio.imread('A/img/image_GS_1_equalized.jpg')

print("IMG Tensor size: "+str(image_blur.shape))

image_borders = borderDetector(image_blur)

iio.imwrite('A/img/image_GS_1_borders.png', image_borders)