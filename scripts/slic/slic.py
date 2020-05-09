import numpy as np
import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import skimage.filters as filters

import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

# SLIC clustering from https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980

#get data to use 
data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
img = data_cube[:,0,:].T
print("max pixel intencity: ", np.max(img))
plt.imshow(img)
#plt.show()
plt.savefig("0.input.jpg")

fig, ax = plt.subplots(1, 1)
ax.hist(img.ravel(), bins=32, range=[0, 1])
ax.set_xlim(0, 1); # 8-bit gives maximum of 256 possible values
#plt.show()
plt.savefig("1.histogram.jpg")
plt.close(fig)

# find number of pixels in image 
num_pixels = img.shape[0]*img.shape[1]
print("number of pixels in org img: ", num_pixels)
regions = num_pixels/20
image_slic = seg.slic(img,n_segments=regions)
plt.imshow(color.label2rgb(image_slic, img, kind='avg'));
#plt.show()
plt.savefig(f'2.slic_{regions/num_pixels}.jpg')
