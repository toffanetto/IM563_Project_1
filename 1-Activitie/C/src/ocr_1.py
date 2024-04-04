# @toffanetto

import numpy as np
import math
import imageio.v2 as iio
import matplotlib.pyplot as plt
from scipy import signal
import skimage as ski

ERODE_KERNEL = 1 # Mask (2*ERODE_KERNEL+1, 2*ERODE_KERNEL+1)

BINARY_THRESHOLD = 150

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

def rgb2GsArray(image_array, r, g, b):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    imageGS = np.zeros([n_rows, n_collumns], dtype=np.uint8) # Empty arraw for the construction of grayscale image

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
        
            intensity = (r/100)*image_array[i, j, 0] + (g/100)*image_array[i, j, 1] + (b/100)*image_array[i, j, 2] # Sum of intensities of RGB Channels

            imageGS[i, j]  = np.uint8(intensity) # Giver for the GS pixel the respective intensity
            
    return imageGS


def binaryImage(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
            if image_array[i, j] < BINARY_THRESHOLD:
                image_array[i, j] = 255
            else:
                image_array[i, j] = 0
        
    return image_array

def erodeImage(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_eroded = np.zeros([n_rows, n_collumns], dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_collumns):
            e = 0
            for n in range(ERODE_KERNEL*2+1):
                if(i-ERODE_KERNEL+n >= 0) and (i-ERODE_KERNEL+n < n_rows):
                    for m in range(ERODE_KERNEL*2+1):
                        if(j-ERODE_KERNEL+m >= 0) and (j-ERODE_KERNEL+m < n_collumns):
                            if(image_array[i,j]>0):
                                e += image_array[i-ERODE_KERNEL+n, j-ERODE_KERNEL+m] # Getting intensity gradient in x-axis
            if(e>=((ERODE_KERNEL*2+1)**2)/2*255):
                image_eroded[i,j] = 255
        print('Eroding image | '+str(np.uint8(i/n_rows*100))+'% completed')

    return image_eroded



######################

try:
    image_gs = iio.imread('C/img/output/image_gs.png')
except:
    image_raw = iio.imread('C/img/text_cutted.jpg')

    image_gs = rgb2GsArray(image_raw, 30, 55, 15)

    iio.imwrite('C/img/output/image_gs.png', image_gs)

image_binary = binaryImage(image_gs)

plt.figure()
plt.imshow(image_binary, cmap='gray')

iio.imwrite('C/img/output/image_binary.png', image_binary)

image_letter_label, letters_count = ski.measure.label(image_binary, connectivity=1, return_num=True)

image_letter_label_color = ski.color.label2rgb(image_letter_label, bg_label=0)

plt.figure()
plt.imshow(image_letter_label_color)
plt.title(str(letters_count)+'/152 words founded')
plt.savefig("./C/plot/image_letter_label_color_1.pdf", format="pdf", bbox_inches="tight")

print(letters_count)

# object_features = ski.measure.regionprops(image_letter_label)
# object_areas = [objf["area"] for objf in object_features]
# print(object_areas)

plt.show()