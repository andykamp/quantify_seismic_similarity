import numpy as np
import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


def perform_and_save_hog(img, o, ppc, cpb, save=True):
    orientations=o
    pixels_per_cell=(ppc, ppc)
    cells_per_block=(cpb, cpb)
    fd, hog_image= hog(img, orientations, pixels_per_cell, cells_per_block,  visualize=True, multichannel=False)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    if save:
        plt.savefig(f'hog_{orientations}_{pixels_per_cell}_{cells_per_block}')
    else:
        plt.show()

if __name__ == "__main__":
    #get data to use 
    data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
    # make sure it is fomrated correctly 
    img = data_cube[:,0,:].T
    print("img shape", img.shape)
    #plt.imshow(img)
    #plt.show()
    perform_and_save_hog(img,8,2,5, save=False)

