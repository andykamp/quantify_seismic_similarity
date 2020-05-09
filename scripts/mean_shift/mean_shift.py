import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from sklearn.cluster import MeanShift, estimate_bandwidth


# get imgage 
#get data to use 
data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
img = data_cube[:,0,:].T
print(img.shape)
image = np.stack((img,)*3, axis=-1)
print(image.shape)
# Shape of original image    
originShape = image.shape


# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg=np.reshape(image, [-1, 3])
print(flatImg.shape)

# Estimate bandwidth for meanshift algorithm    
bandwidth = estimate_bandwidth(flatImg, quantile=0.2, n_samples=500)    
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# Performing meanshift on flatImg    
ms.fit(flatImg)

labels = ms.labels_

# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_    

# Finding and diplaying the number of clusters    
labels_unique = np.unique(labels)    
n_clusters_ = len(labels_unique)    
print("number of estimated clusters : %d" % n_clusters_)    

# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_
print(labels.shape)
plt.figure()
plt.subplot(121), plt.imshow(image), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(np.reshape(labels, image.shape[:2])), plt.axis('off'), plt.title('segmented image with Meanshift', size=20)
#plt.show()
plt.savefig("mean_shift_02_500")