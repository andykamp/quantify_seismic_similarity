import numpy as np
import cv2


""" Fetch from https://handmap.github.io/gradients-and-edge-detection/"""


# Load the image, convert it to grayscale, and show it
image = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Greyscale", image)
cv2.waitKey(0)

# Compute the Laplacian of the image
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
cv2.imwrite("laplacian.png", lap)
cv2.waitKey(0)

# Compute gradients along the X and Y axis, respectively
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

# The sobelX and sobelY images are now of the floating
# point data type -- we need to take care when converting
# back to an 8-bit unsigned integer that we do not miss
# any images due to clipping values outside the range
# of [0, 255]. First, we take the absolute value of the
# graident magnitude images, THEN we convert them back
# to 8-bit unsigned integers
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# We can combine our Sobel gradient images using our
# bitwise OR
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Show our Sobel images
cv2.imshow("Sobel X", sobelX)
cv2.waitKey(0)
cv2.imshow("Sobel Y", sobelY)
cv2.waitKey(0)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.imwrite("sobel_combinded.png", sobelCombined)
cv2.waitKey(0)
