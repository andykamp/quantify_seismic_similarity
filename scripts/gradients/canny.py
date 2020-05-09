import numpy as np
import cv2

""" from https://handmap.github.io/gradients-and-edge-detection/"""
# Load the image, convert it to grayscale, and blur it
# slightly to remove high frequency edges that we aren't
# interested in
image = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0) # different blurs makes it more/less observetnt to small edges
cv2.imshow("Blurred", image)
cv2.imwrite("blurred.png", image)

# When performing Canny edge detection we need two values
# for hysteresis: threshold1 and threshold2. Any gradient
# value larger than threshold2 are considered to be an
# edge. Any value below threshold1 are considered not to
# ben an edge. Values in between threshold1 and threshold2
# are either classified as edges or non-edges based on how
# the intensities are "connected". In this case, any gradient
# values below 30 are considered non-edges whereas any value
# above 150 are considered edges.
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.imwrite("canny-img.png", canny)
cv2.waitKey(0)

for x in range(1, 60, 10):
    for y in range(1, 210, 10):

        canny = cv2.Canny(image, x, y)
        cv2.putText(canny, "x:"+str(x)+" y:"+str(y), (500, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        #cv2.imwrite("_canny-img-"+str(x)+"-"+str(y)+".png", canny)
        cv2.imshow("Canny", canny)
        cv2.waitKey(0)

