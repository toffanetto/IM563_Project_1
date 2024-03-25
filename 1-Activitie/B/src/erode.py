
import cv2
import numpy as np

img = cv2.imread('B/img/output/image_borders.png')


kernel = np.ones((3, 3), np.uint8) 

img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1) 
  
cv2.imshow('Input', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 
  
cv2.waitKey(0) 