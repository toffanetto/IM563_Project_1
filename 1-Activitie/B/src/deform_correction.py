# @toffanetto

import numpy as np
import math
import imageio.v2 as iio
import matplotlib.pyplot as plt
from scipy import signal

ERODE_KERNEL = 1 # Mask (2*ERODE_KERNEL+1, 2*ERODE_KERNEL+1)

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



def imageAxisDraw(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    for j in range(n_collumns): # Collumn sweep
        image_array[np.uint16(n_rows/2), j] = 255
    for i in range(n_rows): # Row sweep
        image_array[i, np.uint16(n_collumns/2)] = 255
        
    return image_array

def binaryImage(image_array):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    threshold = 130

    for j in range(n_collumns): # Collumn sweep
        for i in range(n_rows): # Row sweep
            if image_array[i, j] < threshold:
                image_array[i, j] = 0
            else:
                image_array[i, j] = 255
        
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

def findSquares(image_array):
    squares = []
    
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    K = np.array([[0, 1, 1], 
                   [1, -1, -1], 
                   [1, -1, -1]]) 
    
    mask_sum = 0

    for i in range(n_rows):
        for j in range(n_collumns):
            mask_sum = 0
            for n in range(1*2+1):
                if(i-1+n >= 0) and (i-1+n < n_rows):
                    for m in range(1*2+1):
                        if(j-1+m >= 0) and (j-1+m < n_collumns):
                            mask_sum += K[n,m]*image_array[i-1+n, j-1+m]
            if(mask_sum/255 == 4):
                squares.append((j,i))
    return squares

def distortionCorrrection(image_array, K1, K2, P1, P2):
    n_rows = np.uint16(image_array.shape[0]) # Number of rows of the image
    n_collumns = np.uint16(image_array.shape[1]) # Number of collumns of the image

    image_correct = np.zeros([n_rows, n_collumns], dtype=np.uint8)
    
    xd_ = int((image_array.shape[1])/2)
    yd_ = int((image_array.shape[0])/2)

    for i in range(n_rows):
        for j in range(n_collumns):
            dX = j-xd_
            dY = i-yd_
            r2 = (dX)**2 + (dY)**2
            x1 = dX*r2
            x2 = dX*r2**2
            x3 = r2 + 2*(dX**2)
            x4 = dX*dY*2
            x6 = dY*r2
            x7 = dY*r2**2
            x8 = dX*dY*2
            x9 = r2 + 2*(dY**2)
            
            xu = int(math.floor(dX + x1*K1 + x2*K2 + x3*P1 + x4*P2))
            yu = int(math.floor(dY + x6*K1 + x7*K2 + x8*P1 + x9*P2))
            
            try:
                image_correct[i, j] = image_array[yu, xu]
            except:
                image_correct[i, j] = 120        

    return image_correct

######################

#image_raw = iio.imread('B/img/chess.jpeg')

#image_reduced = imageArrayReduce(image_raw, 25)

#image_gs = rgb2GsArray(image_reduced, 30, 55, 15)

#iio.imwrite('B/img/output/image_gs.png', image_gs)

#image_binary = binaryImage(image_gs)
#image_binary_erode = erodeImage(image_binary)

#iio.imwrite('B/img/output/image_binary_erode.png', image_binary_erode)

image_binary_erode = iio.imread('B/img/output/image_binary_erode.png')

image_original = iio.imread('B/img/image_original.png')

image_original_binary = binaryImage(image_original)

iio.imwrite('B/img/output/image_original_binary.png', image_original_binary)

squares = findSquares(image_binary_erode)

squares_original = findSquares(image_original_binary)

plt.figure()
plt.imshow(image_binary_erode, cmap='gray')
plt.scatter(*zip(*squares))
plt.xlabel('x-axis [pixels]')
plt.ylabel('y-axis [pixels]')
plt.title('Distorted image with markpoints')

plt.savefig("./B/plot/image_binary_erode.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.imshow(image_original_binary, cmap='gray')
plt.scatter(*zip(*squares_original))
plt.xlabel('x-axis [pixels]')
plt.ylabel('y-axis [pixels]')
plt.title('Original image with markpoints')

plt.savefig("./B/plot/image_original_binary.pdf", format="pdf", bbox_inches="tight")

Id = []
Iu = []

xd_ = int((image_original_binary.shape[1])/2)
yd_ = int((image_original_binary.shape[0])/2)

for i in range(len(squares_original)):
    for j in range(len(squares)):
        if np.sqrt((squares_original[i][0]-squares[j][0])**2 + (squares_original[i][1]-squares[j][1])**2) < 20:
            Id.append(squares[j])
            Iu.append(squares_original[i])
            
plt.figure()
Id_marker = plt.scatter(*zip(*Id),marker='o')
Iu_marker = plt.scatter(*zip(*Iu),marker='x')
center = plt.scatter(xd_,yd_,marker='+')
plt.imshow(image_binary_erode, cmap='gray')
plt.legend((Id_marker, Iu_marker, center), ('Distorted position point', 
                                                'Original position point', 'Image center'),fontsize=8)
plt.ylim([80,320])
plt.xlabel('x-axis [pixels]')
plt.ylabel('y-axis [pixels]')
plt.gca().invert_yaxis()
plt.title('Comparison of original and distorted position of markpoints in image')

plt.savefig("./B/plot/markpoints_original_distorted.pdf", format="pdf", bbox_inches="tight")

print(len(Id))
print(len(squares_original))
print(len(squares))


## FIND THE VALUE OF K AND P

# Linear system giver by Y = A*w, where:
#   w = [K1 K2 P1 P2]'

Y = np.zeros([2*len(Id),1])
A = np.zeros([2*len(Id),4])

j = 0

for i in range(0,2*len(Id),2):
    
    dX = Id[j][0]-xd_
    dY = Id[j][1]-yd_
    r2 = (dX)**2 + (dY)**2
    x0 = Iu[j][0] - dX
    x1 = dX*r2
    x2 = dX*r2**2
    x3 = r2 + 2*(dX**2)
    x4 = dX*dY*2
    x5 = Iu[j][1] - dY
    x6 = dY*r2
    x7 = dY*r2**2
    x8 = dX*dY*2
    x9 = r2 + 2*(dY**2)
    
    Y[i,0] = x0
    A[i,0] = x1
    A[i,1] = x2
    A[i,2] = x3
    A[i,3] = x4
    Y[i+1,0] = x5
    A[i+1,0] = x6
    A[i+1,1] = x7
    A[i+1,2] = x8
    A[i+1,3] = x9
    
    j += 1
    
w = np.linalg.pinv(A).dot(Y)

K1 = w[0][0]
K2 = w[1][0]
P1 = w[2][0]
P2 = w[3][0]

print('Y = '+str(Y))

print('A = '+str(A))

print('K1 = '+str(K1)+'\nK2 = '+str(K2)+'\nP1 = '+str(P1)+'\nP2 = '+str(P2))

image_undistorted = distortionCorrrection(image_array=image_binary_erode, K1=K1, K2=K2, P1=P1, P2=P2)

# plt.figure()
# plt.imshow(image_undistorted, cmap='gray')

# iio.imwrite('B/img/output/image_undistorted.png', image_undistorted)


plt.show()