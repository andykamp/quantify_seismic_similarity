import numpy as np
import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import skimage.filters as filters


# get data to use 
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
# manual/supervised treshholding
manual_threshold = 0.5 # value concluded from histogram
img_segmented = img > manual_threshold 
plt.imshow(img_segmented)
#plt.show()
plt.savefig("2.manual_treshold.jpg")


# unsupervised thresholding
unsupervised_threshold = filters.threshold_local(img,block_size=51, offset=0)  # Hit tab with the cursor after the underscore to get all the methods.
img_segmented = img < unsupervised_threshold;
plt.imshow(img_segmented)
#plt.show()
plt.savefig("3.unsupervised_treshold.jpg")
