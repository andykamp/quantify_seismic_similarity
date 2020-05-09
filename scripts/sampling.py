import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks


def plot_patches(patches, num_y, num_x):

    ''' Draws a plot to stich together patches to a image '''
    print(patches.shape, num_y, num_x)
    counter = 1
    for y in range(num_y):
        for x in range(num_x):
            plt.subplot(num_y, num_x, counter)
            plt.imshow(patches[y,x])
            plt.axis('off')
            counter += 1
    plt.show()


# get data to use 
data_cube = np.load("/Users/anderskampenes/Downloads/f3_entire_int8.npy")#"/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
data_cube = data_cube[:,:,:].T

patch_shape = 64
print(data_cube[:,0,:].shape)
#plt.imshow(np.transpose(data_cube[:,:10,:],(1,0,2))[0,:,:])
#plt.show()

pad_x = patch_shape - data_cube.shape[0] % patch_shape
pad_y = patch_shape -  data_cube.shape[-1] % patch_shape
print("inferred paddings", pad_x, pad_y)
seismic_slice = data_cube[:,0,:]
padded_slice = np.pad(seismic_slice, ((0,pad_x),(0, pad_y)), 'edge')
print("padded__slice", padded_slice.shape)
patches = view_as_blocks(padded_slice, (patch_shape, patch_shape))
print("patches", patches.shape)
plot_patches(patches, int(padded_slice.shape[0]/patch_shape), int(padded_slice.shape[1]/patch_shape))