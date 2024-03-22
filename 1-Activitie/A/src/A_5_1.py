# @toffanetto

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

def blurFilter(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_blurred = np.zeros([n_rows, n_collumns, 3], dtype=np.uint8)

    blur_matrix = np.array([[2, 4, 5, 4, 2],
                            [4, 9, 2, 9, 4],
                            [5, 12, 15, 12, 5],
                            [4, 9, 12, 9, 4],
                            [2, 4, 5, 4, 2]])/159
    
    n_filter = np.uint16(blur_matrix.shape[0]) # Number of rows of the filter
    m_filter = np.uint16(blur_matrix.shape[1]) # Number of collumns of the filter

    n_1 = np.int16(np.floor(n_filter/2))
    m_1 = np.int16(np.floor(m_filter/2))
    
    for i in range(n_rows):
        for j in range(n_collumns):
            blur_pixel = 0
            n_pixel = 0
            for n in range(n_filter):
                if(i-n_1+n >= 0) and (i-n_1+n < n_rows):
                    for m in range(m_filter):
                        if(j-m_1+m >= 0) and (j-m_1+m < n_collumns):
                            blur_pixel += image_array[i-n_1+n,j-m_1+m]+blur_matrix[n, m]
                            n_pixel += 1

            image_blurred[i, j] = np.uint8(blur_pixel/n_pixel)
        print(str(np.uint8(i/n_rows*100))+'% completed')

    return image_blurred

image_raw = iio.imread('A/img/image_GS_1_equalized.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

image_blur = blurFilter(image_raw)

iio.imwrite('A/img/image_GS_1_blur.jpg', image_blur)