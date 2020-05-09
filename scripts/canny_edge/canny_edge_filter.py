#import cv2
#im = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/UCSB/1QUARTER/Information_Retrieval_and_Web_Search/ML_WORKSPACE/seismic.png', 0)
#th, bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#edges = cv2.Canny(im, th/2, th)


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png', 0)
print(img.shape)
#img = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/UCSB/1QUARTER/Information_Retrieval_and_Web_Search/ML_WORKSPACE/kmeans.png',0)
#edges = cv2.Canny(img,100,200)
th, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
edges = cv2.Canny(img, th/2, th)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()
plt.savefig('1.input_and_output.jpg')
