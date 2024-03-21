# @toffanetto

import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

HISTOGRAM_THRESHOLD = 0.0004

def getHistogramNorm(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    histogram_array = np.zeros([256]) # Empty arraw for the construction of the histogram

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
        
            histogram_array[image_array[i][j]] += 1
            
    return (histogram_array/n_collumns)/n_rows

def getCumulativeHistogram(histogram_array):

    cumulative_histogram_array = np.zeros([256]) # Empty arraw for the construction of the histogram

    cumulative_histogram_array[0] = histogram_array[0]

    for i in range(len(histogram_array)-1):      
            cumulative_histogram_array[i+1] = cumulative_histogram_array[i] + histogram_array[i+1]
        
            
    return cumulative_histogram_array

def histogramScalling(image_array, histogram):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_equalized = np.zeros([n_rows, n_collumns], dtype=np.uint8)

    scale = -1
    bias = -1

    for i in range(255):
        if (histogram[255-i] > HISTOGRAM_THRESHOLD):
            if scale < 0:
                scale = 255-i
        if (histogram[i] > HISTOGRAM_THRESHOLD):
            if bias < 0:
                bias = i

    print('BIAS factor: '+str(bias))
    print('Scale factor: '+str(scale))

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
            image_equalized[i, j] = min(max((255/scale*image_array[i, j] - bias),0),255)

    return image_equalized
# Read image from file

image_raw = iio.imread('A/img/image_GS_1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

histogram_raw = getHistogramNorm(image_raw)
np.savetxt('./A/data/histogram.txt', histogram_raw, fmt='%.3f', newline=os.linesep)

print("Sum of histogram normalized frequencies: "+str(np.sum(histogram_raw)))

x = np.linspace(0, 255, 256)

plt.figure(figsize = (8,6))
plt.stem(x, histogram_raw)
plt.xlabel('Grayscale 8-bit intensity')
plt.ylabel('Normalized Frequency')
plt.title('Histogram')
plt.grid()

plt.savefig("./A/plot/Histogram.pdf", format="pdf", bbox_inches="tight")

cumulative_histogram_raw = getCumulativeHistogram(histogram_raw)

plt.figure(figsize = (8,6))
plt.plot(x, cumulative_histogram_raw)
plt.xlabel('Grayscale 8-bit intensity')
plt.ylabel('Normalized Cumulative Frequency')
plt.title('Cumulative Histogram')
plt.grid()

plt.savefig("./A/plot/Cumulative_HistogramBeforeScalling.pdf", format="pdf", bbox_inches="tight")

image_equalized = histogramScalling(image_raw, histogram_raw)

histogram_scaled = getHistogramNorm(image_equalized)

plt.figure(figsize = (8,6))
plt.stem(x, histogram_scaled)
plt.xlabel('Grayscale 8-bit intensity')
plt.ylabel('Normalized Frequency')
plt.title('Scaled Histogram')
plt.grid()

plt.savefig("./A/plot/ScaledHistogram.pdf", format="pdf", bbox_inches="tight")


iio.imwrite('A/img/image_GS_1_scale_equalized.jpg', image_equalized)

plt.show()

