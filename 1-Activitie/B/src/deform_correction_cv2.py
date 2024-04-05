# @toffanetto

import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

chessboard_size = (7,7)

objp = np.zeros((np.prod(chessboard_size),3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

object_points = []
image_points = []
image_file = './B/img/output/image_binary_erode.png'

# Load and process each image
image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
# Detect chessboard corners
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
print(ret)
# If corners are found, add object and image points
if ret:
    object_points.append(objp)
    image_points.append(corners)
    # Draw and display the corners
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
    cv2.imshow('img', image)
    cv2.waitKey(500)
    
#cv2.destroyAllWindows()
# Calibrate the camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)
print("Calibration matrix K:\n", K)
print("Distortion:", dist.ravel())

# Load a test image
test_image = cv2.imread(image_file)
# Correct the image distortion
undistorted_image = cv2.undistort(test_image, K, dist, None, K)

# Display the original and corrected image side by side
combined_image = np.hstack((test_image, undistorted_image))
cv2.imshow('Original vs Undistorted', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()