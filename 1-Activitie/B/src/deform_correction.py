# @toffanetto

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

BORDER_THRESHOLD = 14
N_BORDER_SEARCH = 1
M_BORDER_SEARCH = 1

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

def histogramEqualization(image_array, cumulative_histogram):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_equalized = np.zeros([n_rows, n_collumns], dtype=np.uint8)

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
            image_equalized[i, j] = (256-1)*cumulative_histogram[image_array[i, j]]

    return image_equalized

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
        print('BLUR | '+str(np.uint8(i/n_rows*100))+'% completed')

    return image_blurred

def gradVector (Ix, Iy):
    G = np.hypot(Ix,Iy)
    print(G)
    G = np.multiply(np.divide(G,np.max(G)),255)

    theta = np.arctan2(Iy,Ix)
    theta[theta < 0] += np.pi

    return G, theta

def borderDetector(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_borders = np.zeros([n_rows, n_collumns], dtype=np.uint8)
    Ix = np.zeros([n_rows, n_collumns], dtype=np.int32)
    Iy = np.zeros([n_rows, n_collumns], dtype=np.int32)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    for i in range(n_rows):
        for j in range(n_collumns):
            for n in range(N_BORDER_SEARCH*2+1):
                if(i-N_BORDER_SEARCH+n >= 0) and (i-N_BORDER_SEARCH+n < n_rows):
                    for m in range(M_BORDER_SEARCH*2+1):
                        if(j-M_BORDER_SEARCH+m >= 0) and (j-M_BORDER_SEARCH+m < n_collumns):
                            Ix[i,j] += Kx[n,m]*image_array[i-N_BORDER_SEARCH+n, j-M_BORDER_SEARCH+m,0]
                            Iy[i,j] += Ky[n,m]*image_array[i-N_BORDER_SEARCH+n, j-M_BORDER_SEARCH+m,0]

        print('Gradient image | '+str(np.uint8(i/n_rows*100))+'% completed')

    G, theta = gradVector(Ix=Ix, Iy=Iy)

    for i in range(n_rows):
        a = 255
        b = 255
        if(i-N_BORDER_SEARCH >= 0) and (i+N_BORDER_SEARCH < n_rows):
            for j in range(n_collumns):
                    if(j-M_BORDER_SEARCH >= 0) and (j+M_BORDER_SEARCH < n_collumns):

                        if (theta[i,j] >= 0 and theta[i,j] < np.pi/16) or (theta[i,j] >= np.pi/2-np.pi/16 and theta[i,j] < np.pi/2): # E/W
                            a = G[i, j+1]
                            b = G[i, j-1]
                        elif (theta[i,j] >= np.pi/16 and theta[i,j] < np.pi/16+np.pi/8): # NE/SW
                            a = G[i+1, j+1]
                            b = G[i-1, j-1]
                        elif (theta[i,j] >= np.pi/4-np.pi/16 and theta[i,j] < np.pi/4+np.pi/16): #N/S
                            a = G[i+1, j]
                            b = G[i-1, j]
                        elif (theta[i,j] >= np.pi/4+np.pi/16 and theta[i,j] < np.pi/2-np.pi/16): #NW/SE
                            a = G[i-1, j+1]
                            b = G[i-1, j+1]

                        if(G[i,j] >= (max(a,b)) and G[i,j] >= BORDER_THRESHOLD):
                            image_borders[i,j] = 255
            
        print('Thining edges | '+str(np.uint8(i/n_rows*100))+'% completed')

    return image_borders

######################

#image_raw = iio.imread('B/img/squared_deform.jpeg')

#print("IMG Tensor size: "+str(image_raw.shape))

#image_reduced = imageArrayReduce(image_raw, 25)

image_reduced = iio.imread('B/img/output/image_reduced.png')

image_gs= rgb2GsArray(image_reduced, 30, 55, 15)

iio.imwrite('B/img/output/image_gs.png', image_gs)

histogram_raw = getHistogramNorm(image_gs)

cumulative_histogram_raw = getCumulativeHistogram(histogram_raw)

image_equalized = histogramEqualization(image_gs, cumulative_histogram_raw)

iio.imwrite('B/img/output/image_equalized.png', image_equalized)

image_blur = blurFilter(image_equalized)

iio.imwrite('B/img/output/image_blur.png', image_blur)

image_borders = borderDetector(image_blur)

iio.imwrite('B/img/output/image_borders.png', image_borders)