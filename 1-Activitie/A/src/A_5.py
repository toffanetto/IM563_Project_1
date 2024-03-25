# @toffanetto

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

BORDER_THRESHOLD = 10
N_BORDER_SEARCH = 1
M_BORDER_SEARCH = 1

ERODE_KERNEL = 1 # Mask (2*ERODE_KERNEL+1, 2*ERODE_KERNEL+1)

DILATE_KERNEL = 1 # Mask (2*DILATE_KERNEL+1, 2*DILATE_KERNEL+1)

def blurFilter(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_blurred = np.zeros([n_rows, n_collumns, 3], dtype=np.uint8)

    blur_matrix = np.array([[2, 4, 5, 4, 2],
                            [4, 9, 12, 9, 4],
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
        print('Bluring image | '+str(np.uint8(i/n_rows*100))+'% completed')

    return image_blurred


def gradVector (Ix, Iy):
    G = np.hypot(Ix,Iy)
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

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Sobel filter gradient in x-axis 
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # Sobel filter gradient in y-axis

    for i in range(n_rows):
        for j in range(n_collumns):
            for n in range(N_BORDER_SEARCH*2+1):
                if(i-N_BORDER_SEARCH+n >= 0) and (i-N_BORDER_SEARCH+n < n_rows):
                    for m in range(M_BORDER_SEARCH*2+1):
                        if(j-M_BORDER_SEARCH+m >= 0) and (j-M_BORDER_SEARCH+m < n_collumns):
                            Ix[i,j] += Kx[n,m]*image_array[i-N_BORDER_SEARCH+n, j-M_BORDER_SEARCH+m,0] # Getting intensity gradient in x-axis
                            Iy[i,j] += Ky[n,m]*image_array[i-N_BORDER_SEARCH+n, j-M_BORDER_SEARCH+m,0] # Getting intensity gradient in y-axis

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


def dilateImage(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_dilated = np.ones([n_rows, n_collumns], dtype=np.uint8)*255

    for i in range(n_rows):
        for j in range(n_collumns):
            e = 0
            if(image_array[i,j]==0):
                for n in range(DILATE_KERNEL*2+1):
                    if(i-DILATE_KERNEL+n >= 0) and (i-DILATE_KERNEL+n < n_rows):
                        for m in range(DILATE_KERNEL*2+1):
                            if(j-DILATE_KERNEL+m >= 0) and (j-DILATE_KERNEL+m < n_collumns):
                                    e += image_array[i-DILATE_KERNEL+n, j-DILATE_KERNEL+m] # Getting intensity gradient in x-axis
                if(e<((DILATE_KERNEL*2+1)**2)/2*255):
                    image_dilated[i,j] = 0
        print('Dilating image | '+str(np.uint8(i/n_rows*100))+'% completed')

    return image_dilated

try:
    image_blur = iio.imread('A/img/image_GS_1_blur.jpg')

except:
    image_raw = iio.imread('A/img/image_GS_1_equalized.jpg')

    print("IMG Tensor size: "+str(image_raw.shape))

    image_blur = blurFilter(image_raw)

    iio.imwrite('A/img/image_GS_1_blur.jpg', image_blur)

    print("IMG Tensor size: "+str(image_blur.shape))

image_borders = borderDetector(image_blur)

iio.imwrite('A/img/image_GS_1_borders.png', image_borders)

image_borders_erode = erodeImage(image_borders)

iio.imwrite('A/img/image_GS_1_borders_erode.png', image_borders_erode)

image_borders_dilate = dilateImage(image_borders)

iio.imwrite('A/img/image_GS_1_borders_dilate.png', image_borders_dilate)