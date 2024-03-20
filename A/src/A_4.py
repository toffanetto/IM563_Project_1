# @toffanetto

import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

def getHistogram(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    histogram_array = np.zeros([256]) # Empty arraw for the construction of the histogram

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
        
            histogram_array[image_array[i][j][0]] += 1
            
    return (histogram_array/n_collumns)/n_rows

def getCumulativeHistogram(histogram_array):

    cumulative_histogram_array = np.zeros([256]) # Empty arraw for the construction of the histogram

    cumulative_histogram_array[0] = histogram_array[0]

    for i in range(len(histogram_array)-1):      
            cumulative_histogram_array[i+1] = cumulative_histogram_array[i] + histogram_array[i+1]
        
            
    return cumulative_histogram_array

# Read image from file

image_raw = iio.imread('A/img/test_image-1.jpg')

print("IMG Tensor size: "+str(image_raw.shape))

histogram_raw = getHistogram(image_raw)
np.savetxt('./A/data/histogram.txt', histogram_raw, fmt='%.3f', newline=os.linesep)

print(np.sum(histogram_raw))

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

plt.savefig("./A/plot/Cumulative_Histogram.pdf", format="pdf", bbox_inches="tight")

plt.show()

